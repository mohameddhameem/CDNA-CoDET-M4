"""Joern-based CPG generation utilities.

Notes
-----
Runs Joern workflows and prepares raw code property graphs.
"""

import os
import subprocess
import json
import shutil
import hashlib
from multiprocessing import Pool
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from datasets import concatenate_datasets


LANG_EXT = {
    'python': '.py',
    'cpp':    '.cpp',
    'java':   '.java',
}

class JoernRunner:
    def __init__(self, temp_dir="./temp_joern", joern_path=None):
        """
        Initialize runner with a temporary directory for intermediate processing.
        """
        self.temp_dir = os.path.abspath(temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Set paths for Joern tools
        if joern_path:
            self.joern_parse = os.path.join(joern_path, "joern-parse")
            self.joern_export = os.path.join(joern_path, "joern-export")
        else:
            self.joern_parse = "joern-parse"
            self.joern_export = "joern-export"

    def parse_one(self, idx, code_string, language='python'):
        """
        Process a single code snippet.
        Returns a dictionary: {'idx': int, 'graphml': str}
        """
        # Calculate MD5 hash of the code string
        code_hash = hashlib.md5(code_string.encode('utf-8')).hexdigest()
        
        # Create unique temp filenames based on index to avoid collisions
        uid = f"sample_{idx}"
        src_file = os.path.join(self.temp_dir, f"{uid}{LANG_EXT.get(language, '.py')}")
        cpg_bin = os.path.join(self.temp_dir, f"{uid}.bin")
        out_dir = os.path.join(self.temp_dir, f"out_{uid}")
        graphml_path = os.path.join(out_dir, "export.xml")

        graphml_content = None

        try:
            # 1. Write source code (RAW CODE)
            with open(src_file, "w", encoding='utf-8') as f:
                f.write(code_string)

            # 2. Joern Parse: Generate CPG binary
            subprocess.run(
                [self.joern_parse, src_file, "-o", cpg_bin],
                check=True, capture_output=True, text=True
            )

            # 3. Joern Export: Convert CPG to GraphML
            subprocess.run(
                [self.joern_export, cpg_bin, "-o", out_dir, "--repr", "all", "--format", "graphml"],
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

        return {'idx': idx, 'code_hash': code_hash, 'graphml': graphml_content, 'language': language, 'code': code_string}

def worker_func(args):
    """Unpack arguments for the worker."""
    runner, idx, code, language = args
    return runner.parse_one(idx, code, language)


def small_sample(data, seed=42):
    sampled_splits = []
    languages = set(data['language'])
    
    for lang in languages:
        lang_data = data.filter(lambda x: x['language'] == lang).shuffle(seed=seed)
        
        targets = {
            'train': 20000,
            'val': 2000,
        }
        
        for split, target_count in targets.items():
            split_data = lang_data.filter(lambda x: x['split'] == split).shuffle(seed=seed)
            models = set(split_data['model'])
            
            if len(models) == 0:
                continue
            
            target_per_model = target_count // len(models)
            split_samples = []
            
            for model in models:
                model_data = split_data.filter(lambda x: x['model'] == model).shuffle(seed=seed)
                sampled = stratified_by_source(model_data, target_per_model, seed)
                split_samples.append(sampled)
            
            if split_samples:
                sampled_splits.append(concatenate_datasets(split_samples))

        test_data = lang_data.filter(lambda x: x['split'] == 'test')
        if len(test_data) > 0:
            sampled_splits.append(test_data)
    
    return concatenate_datasets(sampled_splits)


def stratified_by_source(data, target_n, seed=42):
    sources = set(data['source'])
    total = len(data)
    
    if total <= target_n:
        return data
    
    selected = []
    for source in sources:
        source_data = data.filter(lambda x: x['source'] == source) 
        n = max(1, round(target_n * len(source_data) / total))
        # Select front N since data is already shuffled
        selected.append(source_data.select(range(min(n, len(source_data)))))
    
    result = concatenate_datasets(selected)
    
    if len(result) > target_n:
        result = result.select(range(target_n))
    
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--path', type=str, default="CPG")
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--small_sample', action='store_true', help="Whether to use a small stratified sample for testing")
    parser.add_argument('--obfuscate', action='store_true', help="Whether to obfuscate code before processing (not implemented in this script, placeholder for future use)")
    args = parser.parse_args()

    # 1. Load Dataset
    print("Loading Dataset...")
    dataset = load_dataset("DaniilOr/CoDET-M4")
    
    # Filter to only include samples where 'model' is not None
    data = dataset['train'].filter(lambda x: x['model'] != None and x['language'] != 'cpp')
        
    if args.small_sample:
        data = small_sample(data)
    
    # Apply limit if specified (for testing)
    if args.limit:
        data = data.select(range(min(args.limit, len(data))))

    # 2. Configuration
    output_file = f"./{args.path}/raw/cpg_dataset.jsonl"  # Changed filename to indicate raw code
    joern_path = '/home/shenghua/bin/joern-cli/' # Update with your path
    temp_dir = f"./{args.path}/temp_joern_workers"
    
    # Create necessary directories
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize runner configuration
    runner = JoernRunner(temp_dir=temp_dir, joern_path=joern_path)

    if args.obfuscate:
        from utils.obfuscator import obfuscate_python, obfuscate_java
        # infer only test split for obfuscation to save time, since train/val are not used for evaluation
        data = data.filter(lambda x: x['split'] == 'test')
        _OBFUSCATORS = {
            'python': obfuscate_python,
            'java':   obfuscate_java,
        }

        def get_code(item):
            """Return obfuscated code if the language is supported, else raw code."""
            fn = _OBFUSCATORS.get(item['language'])
            if fn is None:
                return item['cleaned_code']
            try:
                obfuscated, _ = fn(item['cleaned_code'])
                return obfuscated
            except Exception:
                # Fall back to raw code if obfuscation fails
                return item['cleaned_code']

        print("Preparing arguments (obfuscated code)...")
    else:
        def get_code(item):
            return item['code']

        print("Preparing arguments (using RAW 'code' column)...")
        
    # 3. Prepare Arguments
    # === KEY CHANGE HERE: Using item['code'] instead of item['cleaned_code'] ===
    process_args = [
        (runner, i, get_code(item), item['language'])
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
                    'code': result['code'], 
                    'graphml': result['graphml']
                }
                
                # Write to JSONL
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Cleanup temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    print(f"Done! All raw-code based CPGs saved to {output_file}")