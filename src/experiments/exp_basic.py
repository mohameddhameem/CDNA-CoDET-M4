import os
import random
import numpy as np
import torch
import torch.distributed as dist
from src.models.codeclip import CodeCLIP, CodeCLIP_ablation_graph_only, CodeCLIP_ablation_no_penalty, \
    CodeCLIP_ablation_no_router, CodeBert, Unixcoder
from datetime import datetime
from src.experiments.root import ROOT_DIR


class ExpBasic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "full": CodeCLIP,
            "graph_only": CodeCLIP_ablation_graph_only,
            "codebert":CodeBert,
            "unixcoder":Unixcoder,
            "no_router":CodeCLIP_ablation_no_router,
            "no_penalty":CodeCLIP_ablation_no_penalty,
        }
       
        if self.args.task_name == "pretrain":
            self.setting = self._pretrain_setting()
        else:
            self.setting = self._setting()

        self._set_seed(args.seed)
        self._setup_logger()

    @staticmethod
    def _is_main_process():
        """Check if current process is the main process (rank 0)"""
        return not dist.is_initialized() or dist.get_rank() == 0

    def _print_main(self, *args, **kwargs):
        """Print only in main process"""
        if self._is_main_process():
            print(*args, **kwargs)

    def _setting(self):
        setting = "{}({})_{}({})".format(
            self.args.model,
            self.args.task_name,
            self.args.language, # CPP, Java, Python
            self.args.seed,
        )
        return setting
    
    def _pretrain_setting(self):
        setting = "{}({})_{}".format(
            self.args.model,
            "pretrain",
            self.args.language, # CPP, Java, Python
        )
        return setting

    def _setup_logger(self):
        """Setup simple logger - creates log file in logs directory"""
        if not self._is_main_process() or self.args.is_logging is False:
            self.log_file = None
            return

        # Create logs directory
        log_dir = os.path.join(ROOT_DIR, "logs", self.args.model)
        os.makedirs(log_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.setting}_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)
        self.log_file = open(log_path, "w", encoding="utf-8")
        self._write_log(f"Experiment started: {self.args.model}")
        self._write_log(f"Experiment setting: {self.setting}")
        print(f"Log file created: {log_path}")

    def _write_log(self, message: str):
        """Write a message to the log file if it exists"""
        if (
            hasattr(self, "log_file")
            and self._is_main_process()
            and self.args.is_logging is True
        ):
            self.log_file.write(f"[{datetime.now()}] {message}\n")
            self.log_file.flush()

    def __del__(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "log_file") and self.log_file is not None:
            self.log_file.close()
        torch.cuda.empty_cache()

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        self._print_main(f"Set random seed to {seed}")

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.gpu_type == "cuda" and torch.cuda.is_available():
                if self.args.use_multi_gpu:
                    os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                    if "LOCAL_RANK" in os.environ:
                        local_rank = int(os.environ["LOCAL_RANK"])
                    else:
                        raise ValueError(
                            "LOCAL_RANK environment variable is not set for multi-GPU training."
                        )
                    torch.cuda.set_device(local_rank)
                    if not dist.is_initialized():
                        dist.init_process_group(backend="nccl")
                    self._print_main(f"Use multi-GPU: {self.args.devices}")
                    return torch.device(local_rank)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                    print(f"Use GPU: cuda {self.args.gpu}")
                    return torch.device(f"cuda:{self.args.gpu}")
            elif self.args.gpu_type == "mps" and torch.backends.mps.is_available():
                print("Use GPU: mps")
                return torch.device("mps")

        print("Use CPU")
        return torch.device("cpu")

    def _load_checkpoint(self, model_path, model, strict=True):
        """Load checkpoint with proper handling of DDP wrapper"""

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle DDP wrapper key mismatch
            if hasattr(model, "module"):
                # Current model is DDP wrapped, but checkpoint might not have 'module.' prefix
                if not any(key.startswith("module.") for key in checkpoint.keys()):
                    # Add 'module.' prefix to checkpoint keys
                    checkpoint = {f"module.{k}": v for k, v in checkpoint.items()}
            else:
                # Current model is not DDP wrapped, but checkpoint might have 'module.' prefix
                if any(key.startswith("module.") for key in checkpoint.keys()):
                    # Remove 'module.' prefix from checkpoint keys
                    checkpoint = {
                        k.replace("module.", ""): v for k, v in checkpoint.items()
                    }

            model.load_state_dict(checkpoint, strict=strict)
            self._print_main(f"Loaded checkpoint from {model_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    def _monitor_resources(self, shm_threshold=75, gpu_threshold=80, auto_gc=True, verbose=True):
        """
        Monitor /dev/shm, CPU memory, and GPU memory usage.
        Supports single GPU, DataParallel (DP), and DistributedDataParallel (DDP).
        Automatically triggers cleanup if usage exceeds thresholds.
        """
        import shutil
        import psutil
        # --- /dev/shm ---
        total, used, free = shutil.disk_usage("/dev/shm")
        shm_percent = used / total * 100
        # --- CPU memory ---
        mem = psutil.virtual_memory()
        if verbose:
            self._print_main(
                f"\n[Monitor] /dev/shm: {shm_percent:.2f}% ({used // (1024**3)}G / {total // (1024**3)}G)"
            )
            self._print_main(
                f"[Monitor] CPU memory: {mem.percent:.2f}% ({mem.used // (1024**3)}G / {mem.total // (1024**3)}G)"
            )

        # --- GPU memory ---
        gpu_percent_max = 0.0
        if torch.cuda.is_available() and hasattr(self, "device") and self.device.type == "cuda":
            num_devices = torch.cuda.device_count()

            for d in range(num_devices):
                mem_alloc = torch.cuda.memory_allocated(d) / 1024**2
                mem_total = torch.cuda.get_device_properties(d).total_memory / 1024**2
                percent = (mem_alloc / mem_total * 100) if mem_total > 0 else 0.0
                gpu_percent_max = max(gpu_percent_max, percent)
                if verbose:
                    self._print_main(
                        f"[Monitor] GPU {d} memory: {percent:.2f}% ({mem_alloc:.0f}MB / {mem_total:.0f}MB)"
                    )

        # --- Trigger cleanup if thresholds exceeded ---
        if auto_gc and (shm_percent > shm_threshold or mem.percent > 90):
            self._print_main(
                "/dev/shm or CPU memory usage is high, triggering CPU memory cleanup"
            )
            self._clear_cpu_memory()

        if auto_gc and gpu_percent_max > gpu_threshold:
            self._print_main("GPU memory usage is high, triggering GPU memory cleanup")
            self._clear_gpu_memory()

    def _clear_cpu_memory(self):
        """Force Python CPU memory cleanup using garbage collector"""
        import gc

        gc.collect()
        self._print_main("CPU memory cleared")

    def _clear_gpu_memory(self):
        """Force GPU memory cleanup by releasing unused cached memory, safe for DP/DDP"""
        torch.cuda.empty_cache()
        self._print_main("GPU memory cleared")

    def get_loader(self):
        raise NotImplementedError

    def _build_model(self):
        raise NotImplementedError

    def _get_pretrain_model(self):
        raise NotImplementedError

    def train(self):
        pass

    def vali(self):
        pass

    def test(self):
        pass