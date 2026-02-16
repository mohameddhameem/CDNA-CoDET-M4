import os
import subprocess
import json
import shutil
from multiprocessing import Pool
from datasets import load_dataset
from tqdm import tqdm

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

    def parse_one(self, idx, code_string):
        """
        Process a single code snippet. ONLY Python
        Returns a dictionary: {'idx': int, 'graphml': str}
        """
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
            return {'idx': idx, 'graphml': None, 'error': str(e)}
        
        finally:
            # Cleanup temp files immediately to save space/inodes
            if os.path.exists(src_file): os.remove(src_file)
            if os.path.exists(cpg_bin): os.remove(cpg_bin)
            if os.path.exists(out_dir): shutil.rmtree(out_dir, ignore_errors=True)

        return {'idx': idx, 'graphml': graphml_content}

def worker_func(args):
    """Unpack arguments for the worker."""
    runner, idx, code = args
    return runner.parse_one(idx, code)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    # 1. Load Dataset
    print("Loading Dataset...")
    dataset = load_dataset("DaniilOr/CoDET-M4")
    
    # Filter for Python code only
    py_data = dataset['train'].filter(lambda x: x['language'] == 'python')
    
    # Apply limit if specified (for testing)
    if args.limit:
        py_data = py_data.select(range(min(args.limit, len(py_data))))

    # 2. Configuration
    output_file = "./CPG/raw/cpg_dataset.jsonl"  # Changed filename to indicate raw code
    joern_path = '/home/shenghua/bin/joern-cli/' # Update with your path
    temp_dir = "./CPG/temp_joern_workers"
    
    # Create necessary directories
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize runner configuration
    runner = JoernRunner(temp_dir=temp_dir, joern_path=joern_path)

    # 3. Prepare Arguments
    # === KEY CHANGE HERE: Using item['code'] instead of item['cleaned_code'] ===
    print("Preparing arguments (using RAW 'code' column)...")
    process_args = [
        (runner, i, item['code']) 
        for i, item in enumerate(py_data)
    ]#[:10]

    print(f"Starting processing for {len(process_args)} items with {args.workers} workers...")
    
    # 4. Processing and Writing
    with open(output_file, "w", encoding='utf-8') as f_out:
        with Pool(args.workers) as pool:
            # Using imap_unordered for speed
            for result in tqdm(pool.imap_unordered(worker_func, process_args), total=len(process_args)):
                
                # Retrieve original item to get labels
                original_item = py_data[result['idx']]
                
                record = {
                    'idx': result['idx'],           # Global index for alignment
                    'target': original_item['target'],
                    'model': original_item['model'],
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
