import os
import subprocess
import uuid
import hashlib
from multiprocessing import Pool, cpu_count
from functools import partial

class JoernRunner:
    def __init__(self, output_dir="./CPG", joern_path=None):
        """
        Initialize Joern runner
        :param output_dir: Directory to store output files
        :param joern_path: Path to joern-cli directory (None if in PATH)
        """
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set command paths
        if joern_path:
            self.joern_parse = os.path.join(joern_path, "joern-parse")
            self.joern_export = os.path.join(joern_path, "joern-export")
        else:
            self.joern_parse = "joern-parse"
            self.joern_export = "joern-export"

    def parse_string(self, code_string, language="c", skip_existing=True):
        """
        Parse code string and generate GraphML
        :param code_string: Source code to parse
        :param language: Programming language (c, cpp, python, java, js, php, go)
        :param skip_existing: Skip if file already exists
        :return: Path to generated GraphML file or None if failed
        """
        # Map language to file extension
        ext_map = {
            'c': '.c', 'cpp': '.cpp', 'python': '.py', 'java': '.java',
            'js': '.js', 'php': '.php', 'go': '.go'
        }
        ext = ext_map.get(language.lower(), '.c')
        
        # Generate deterministic file names based on code hash
        code_hash = hashlib.md5(code_string.encode('utf-8')).hexdigest()[:8]
        uid = code_hash
        src_file = os.path.join(self.output_dir, f"temp_{uid}{ext}")
        cpg_file = os.path.join(self.output_dir, f"cpg_{uid}.bin")
        out_file = os.path.join(self.output_dir, f"{uid}.graphml")
        code_file = os.path.join(self.output_dir, f"{uid}.txt")
        out_dir = os.path.join(self.output_dir, f"out_{uid}")

        # Skip if already exists
        if skip_existing and os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            return out_file

        try:
            # Write source code to file
            with open(src_file, "w", encoding='utf-8') as f:
                f.write(code_string)
            
            # Save original code text
            with open(code_file, "w", encoding='utf-8') as f:
                f.write(code_string)

            # Generate CPG
            subprocess.run(
                [self.joern_parse, src_file, "-o", cpg_file],
                check=True,
                capture_output=True,
                text=True
            )

            # Export to GraphML (outputs to out_dir/export.xml)
            subprocess.run(
                [self.joern_export, cpg_file, "-o", out_dir, "--repr", "all", "--format", "graphml"],
                check=True,
                capture_output=True,
                text=True
            )

            # Move export.xml to final location
            export_file = os.path.join(out_dir, "export.xml")
            if os.path.exists(export_file):
                os.rename(export_file, out_file)
                os.rmdir(out_dir)

            # Verify output
            if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                print(f"Success: {out_file}")
                return out_file
            else:
                print("Error: Empty output file")
                return None

        except subprocess.CalledProcessError as e:
            print(f"Joern error: {e.stderr}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
        finally:
            # Cleanup temporary files
            for f in [src_file, cpg_file]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
            # Cleanup temporary output directory if it still exists
            if os.path.exists(out_dir):
                try:
                    import shutil
                    shutil.rmtree(out_dir)
                except OSError:
                    pass


def process_code_wrapper(args):
    """Wrapper function for multiprocessing"""
    code, output_dir, joern_path, language, skip_existing = args
    runner = JoernRunner(output_dir=output_dir, joern_path=joern_path)
    return runner.parse_string(code, language=language, skip_existing=skip_existing)


if __name__ == "__main__":
    import argparse
    from datasets import load_dataset
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Generate CPG with Joern')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of code snippets')
    args = parser.parse_args()

    # Load the CoDET-M4 dataset
    print("Loading dataset...")
    dataset = load_dataset("DaniilOr/CoDET-M4")
    python_dataset = dataset['train'].filter(lambda x: x['language'] == 'python')
    
    if args.limit:
        python_dataset = python_dataset.select(range(min(args.limit, len(python_dataset))))
    
    output_dir = './CPG/graphml'
    joern_path = '/home/shenghua/bin/joern-cli/'
    language = 'python'
    
    print(f"Processing {len(python_dataset)} Python code snippets with {args.workers} workers...")
    print("💡 Press Ctrl+C to stop (already processed files will be skipped on restart)\n")
    
    # Prepare arguments for parallel processing
    process_args = [(code, output_dir, joern_path, language, True) 
                    for code in python_dataset['code']]
    
    # Process in parallel
    success = 0
    failed = 0
    skipped = 0
    
    try:
        with Pool(processes=args.workers) as pool:
            for result in tqdm(pool.imap_unordered(process_code_wrapper, process_args), 
                             total=len(process_args), desc="Parsing code"):
                if result:
                    if os.path.exists(result) and os.path.getsize(result) > 0:
                        success += 1
                    else:
                        skipped += 1
                else:
                    failed += 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Progress saved. Run again to resume.")
    
    print(f"\n{'='*60}")
    print(f"✅ Success: {success} | ⏭️  Skipped: {skipped} | ❌ Failed: {failed}")
    print(f"Total: {success + skipped + failed}")
    print(f"{'='*60}")