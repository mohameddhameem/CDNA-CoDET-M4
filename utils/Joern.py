import os
import subprocess
import json
import shutil
import hashlib
from multiprocessing import Pool
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

class JoernRunner:
    def __init__(self, temp_dir="./temp_joern", joern_path='/storage/home/dhameem.m.2025/bin/joern/joern-cli'):
        """
        Initialize runner with a temporary directory for intermediate processing.
        Uses java -cp CLI invocation instead of shell wrappers to avoid Windows path issues.
        """
        self.temp_dir = os.path.abspath(temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

        # Set paths for Joern tools using java -cp instead of shell wrappers
        if joern_path:
            lib_path = os.path.join(joern_path, "lib", "*")
            self.joern_parse = ["java", "-cp", lib_path, "io.joern.joerncli.JoernParse"]
            self.joern_export = ["java", "-cp", lib_path, "io.joern.joerncli.JoernExport"]
        else:
            # Fallback to PATH lookup (less reliable on Windows)
            self.joern_parse = "joern-parse"
            self.joern_export = "joern-export"

    def parse_one(self, idx, code_string):
        """
        Process a single code snippet. ONLY Python
        Returns a dictionary: {'idx': int, 'graphml': str}
        """
        # Calculate MD5 hash of the code string
        code_hash = hashlib.md5(code_string.encode('utf-8')).hexdigest()

        # Create unique temp filenames based on index to avoid collisions
        uid = f"sample_{idx}"
        src_file = os.path.join(self.temp_dir, f"{uid}.py")
        cpg_bin = os.path.join(self.temp_dir, f"{uid}.bin")
        out_dir = os.path.join(self.temp_dir, f"out_{uid}")
        graphml_path = os.path.join(out_dir, "export.xml")

        graphml_content = None

        try:
            # 1. Write source code (RAW CODE)
            with open(src_file, "w", encoding='utf-8') as f:
                f.write(code_string)

            # 2. Joern Parse: Generate CPG binary
            parse_cmd = self.joern_parse if isinstance(self.joern_parse, list) else [self.joern_parse]
            subprocess.run(
                parse_cmd + [src_file, "-o", cpg_bin],
                check=True, capture_output=True, text=True
            )

            # 3. Joern Export: Convert CPG to GraphML
            export_cmd = self.joern_export if isinstance(self.joern_export, list) else [self.joern_export]
            subprocess.run(
                export_cmd + [cpg_bin, "-o", out_dir, "--repr", "all", "--format", "graphml"],
                check=True, capture_output=True, text=True
            )

            # 4. Read the generated GraphML content
            if os.path.exists(graphml_path):
                with open(graphml_path, 'r', encoding='utf-8') as f:
                    graphml_content = f.read()

        except Exception as e:
            # Return None for graphml if failed, but keep idx
            return {'idx': idx, 'code_hash': code_hash, 'graphml': None, 'error': str(e)}

        finally:
            # Cleanup temp files immediately to save space/inodes
            if os.path.exists(src_file): os.remove(src_file)
            if os.path.exists(cpg_bin): os.remove(cpg_bin)
            if os.path.exists(out_dir): shutil.rmtree(out_dir, ignore_errors=True)

        return {'idx': idx, 'code_hash': code_hash, 'graphml': graphml_content}

def worker_func(args):
    """Unpack arguments for the worker."""
    runner, idx, code = args
    return runner.parse_one(idx, code)

def small_sample(data):
    # Stratified sampling by language, split, and model
    sampled_splits = []
    languages = set(data['language'])
    
    for lang in languages:
        lang_data = data.filter(lambda x: x['language'] == lang)
        
        # Define target counts for each split
        targets = {
            'train': 2400,
            'val': 300,
            'test': 300
        }
        
        for split, target_count in targets.items():
            split_data = lang_data.filter(lambda x: x['split'] == split)
            models = set(split_data['model'])
            
            if len(models) == 0:
                continue
                
            # Calculate target per model
            target_per_model = target_count // len(models)
            
            split_samples = []
            for model in models:
                model_data = split_data.filter(lambda x: x['model'] == model)
                if len(model_data) > target_per_model:
                     split_samples.append(model_data.select(range(target_per_model)))
                else:
                     split_samples.append(model_data)
            
            if split_samples:
                sampled_splits.append(concatenate_datasets(split_samples))
        
    return concatenate_datasets(sampled_splits)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=15)
    parser.add_argument('--path', type=str, default="CPG", help='Path to CPG directory')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples for testing')
    parser.add_argument('--joern-path', type=str, default=None, help='Path to joern-cli directory. If not provided, uses JOERN_PATH env var or default')
    parser.add_argument('--small_sample', action='store_true', help="Whether to use a small stratified sample for testing")
    args = parser.parse_args()

    # 1. Load Dataset
    print("Loading Dataset...")
    dataset = load_dataset("DaniilOr/CoDET-M4")

    # Filter to only include samples where 'model' is not None
    data = dataset['train'].filter(lambda x: x['model'] != None)

    if args.small_sample:
        data = small_sample(data)

    # Apply limit if specified (for testing)
    if args.limit:
        data = data.select(range(min(args.limit, len(data))))

    # 2. Configuration
    output_file = f"./{args.path}/raw/cpg_dataset.jsonl"  # Changed filename to indicate raw code
    # Determine joern_path: CLI arg > env var > default
    joern_path = args.joern_path or os.environ.get('JOERN_PATH', r'C:\Learning\SMU\City-of-Agents-1\joern-cli')
    if not os.path.exists(joern_path):
        raise FileNotFoundError(f"Joern path not found: {joern_path}. Set --joern-path or JOERN_PATH env var")
    temp_dir = f"./{args.path}/temp_joern_workers"
    
    # Create necessary directories
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize runner configuration
    runner = JoernRunner(temp_dir=temp_dir, joern_path=joern_path)

    # 3. Prepare Arguments
    # === KEY CHANGE HERE: Using item['code'] instead of item['cleaned_code'] ===
    print("Preparing arguments (using RAW 'code' column)...")
    process_args = [
        (runner, i, item['code']) 
        for i, item in enumerate(data)
    ]

    print(f"Starting processing for {len(process_args)} items with {args.workers} workers...")
    
    # 4. Processing and Writing
    with open(output_file, "w", encoding='utf-8') as f_out:
        with Pool(args.workers) as pool:
            # Using imap_unordered for speed
            for result in tqdm(pool.imap_unordered(worker_func, process_args), total=len(process_args)):
                
                # Retrieve original item to get labels
                original_item = data[result['idx']]
                record = {
                    'idx': result['idx'],  # Keep original index just in case
                    'hash': result['code_hash'],     # Use hash as unique identifier
                    'target': original_item['target'],
                    'model': original_item['model'],
                    'language': original_item['language'],
                    'split': original_item['split'],
                    'source': original_item['source'],
                    # Optional: Store the raw code in JSON for reference
                    'code': original_item['code'], 
                    'graphml': result['graphml']
                }
                
                # Write to JSONL
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Cleanup temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    print(f"Done! All raw-code based CPGs saved to {output_file}")