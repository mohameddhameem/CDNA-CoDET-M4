from exp.exp_basic import ExpBasic
import torch
from torch import optim
from torch.nn import DataParallel
from torch_geometric.loader import DataListLoader, DataLoader
from utils.tools import EarlyStopping
import os
import time
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class ExpDownstream(ExpBasic):
    def __init__(self, args, dataset):
        super(ExpDownstream, self).__init__(args)
        # dataset can be a dict of DataLoader or a dataset
        self.dataset = dataset
        self.train_loader, self.val_loader, self.test_loader = self._build_loaders()
        self._print_main(
            "Downstream split sizes: "
            f"train={len(self.train_loader.dataset) if self.train_loader else 0}, "
            f"val={len(self.val_loader.dataset) if self.val_loader else 0}, "
            f"test={len(self.test_loader.dataset) if self.test_loader else 0}"
        )

        # Device and model
        self.device = self._acquire_device()
        self.args.metadata = (self.dataset[0].node_types, self.dataset[0].edge_types)
        self.classes = set(self.dataset.data.model)
        self.args.num_classes = len(self.classes)
        self._print_main(f"Classes: {self.classes}")
        self._write_log(f"Classes: {self.classes}")
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.classes))}

        if self.args.pattern == "pretrain":
            self.pre_model = self._get_pretrain_model()
        else:
            self.pre_model = None

        self.model = self._build_model().to(self.device)
        # if self.args.use_multi_gpu and torch.cuda.device_count() > 1:
        #     self.model = DataParallel(self.model)

        # Optimizer / criterion / early stop
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.checkpoint_path = self.args.checkpoints + "/" + self.setting
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

    def _get_pretrain_model(self):
        self.pretrain_setting = self._pretrain_setting()
        path = (
                self.args.checkpoints + "/" + self.pretrain_setting + "/" + "checkpoint.pth"
        )
        pretrain_model = (
            self.model_dict[self.args.model].Pretrain(self.args).float()
        )
        self._load_checkpoint(path, pretrain_model)
        return pretrain_model

    def _build_model(self):
        model = self.model_dict[self.args.model].Downstream(self.pre_model, self.args).float()
        return model
    
    def _build_loaders(self):
        train_lst = self.dataset.get_subset(language=self.args.language, split="train")
        val_lst = self.dataset.get_subset(language=self.args.language, split="val")
        test_lst = self.dataset.get_subset(language=self.args.language, split="test")

        train_loader = DataLoader(train_lst, batch_size=self.args.batch_size, shuffle=True) if train_lst else None
        val_loader = DataLoader(val_lst, batch_size=self.args.batch_size, shuffle=False) if val_lst else None
        test_loader = DataLoader(test_lst, batch_size=self.args.batch_size, shuffle=False) if test_lst else None

        return train_loader, val_loader, test_loader

    def _run_epoch(self, loader, epoch, training=True):
        if training:
            self.model.train()
        else:
            self.model.eval()

        epoch_loss = 0.0
        total = 0

        iterator = tqdm(loader, desc=f"Epoch {epoch+1} {'train' if training else 'eval'}")
        for batch in iterator:
            batch = batch.to(self.device)

            if training:
                self.optimizer.zero_grad()

            logits = self.model(batch)
            labels = torch.tensor([self.label_map[label] for label in batch.model], device=self.device)

            if training:
                loss_total = self.criterion(logits, labels)
                loss_total.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    loss_total = self.criterion(logits, labels)

            batch_size = len(batch)
            total += batch_size
            epoch_loss += float(loss_total) * batch_size
            if self._is_main_process() and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    {"loss": f"{loss_total.item():.6f}", 
                        "avg_loss": f"{epoch_loss / max(total, 1):.6f}"}
                )
        avg_loss = epoch_loss / max(total, 1)
        return avg_loss

    def train(self):
        if self.args.continue_train is True:
            model_path = os.path.join(self.checkpoint_path, "checkpoint.pth")
            self._load_checkpoint(model_path, self.model)

        time_start = time.time()
        for epoch in range(self.args.epochs):
            start = time.time()
            train_loss = self._run_epoch(self.train_loader, epoch, training=True)
            val_loss = train_loss
            if self.val_loader is not None:
                with torch.no_grad():
                    val_loss = self._run_epoch(self.val_loader, epoch, training=False)

            self._print_main(
                f"Epoch {epoch+1}/{self.args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={time.time()-start:.1f}s"
            )
            self._write_log(
                f"Epoch {epoch+1}/{self.args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={time.time()-start:.1f}s"
            )

            self.early_stopping(val_loss, self.model, path=self.checkpoint_path)
            if self.early_stopping.early_stop:
                self._print_main("Early stopping triggered")
                self._write_log("Early stopping triggered")
                break

            self.test(reload=False)

        training_info = {
            "best_loss": self.early_stopping.val_loss_min,
            "training_time": time.time() - time_start,
            "model_path": self.checkpoint_path + "/checkpoint.pth",
        }

        self._print_main(
            f"\nTraining finished:",
            f'Best loss: {training_info["best_loss"]:.6f} ',
            f'Training time: {training_info["training_time"]:.2f}s',
            f'\nSave path: {training_info["model_path"]}',
        )
        self._write_log(
            f"Training finished:"
            f'Best loss: {training_info["best_loss"]:.6f} '
            f'Training time: {training_info["training_time"]:.2f}s\n'
            f'Save path: {training_info["model_path"]}'
        )
        self.test(reload=True)


    def test(self, reload=True):
        """Evaluate model on test/validation set and return P, R, F1, Accuracy."""
        self.model.eval()
        all_preds = []
        all_labels = []
        if reload:
            best_model_path = os.path.join(self.checkpoint_path, "checkpoint.pth")
            self._load_checkpoint(best_model_path, self.model)
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                preds = torch.argmax(logits, dim=1)
                labels = torch.tensor([self.label_map[label] for label in batch.model], device=self.device)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)

        self._print_main(f"Test Results -> P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}, A: {accuracy:.4f}")
        self._write_log(f"Test Results -> P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}, A: {accuracy:.4f}")
    
    