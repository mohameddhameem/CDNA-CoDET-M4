#!/usr/bin/env python3
"""
Debug script to test Joern parsing directly
"""
import subprocess
import os
import tempfile
import shutil

JOERN_PATH = '/storage/home/dhameem.m.2025/bin/joern/joern-cli'

# Test Java installation
print("=" * 60)
print("1. Testing Java...")
print("=" * 60)
try:
    result = subprocess.run(['java', '-version'], capture_output=True, text=True, timeout=5)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return code:", result.returncode)
except Exception as e:
    print(f"ERROR: {e}")

# Test Joern parse command
print("\n" + "=" * 60)
print("2. Testing Joern Parse...")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    # Create a simple Python file
    test_py = os.path.join(tmpdir, 'test.py')
    with open(test_py, 'w') as f:
        f.write('def hello():\n    print("hello")\n')
    
    print(f"Created test file: {test_py}")
    
    # Build parse command
    lib_path = os.path.join(JOERN_PATH, "lib", "*")
    parse_cmd = ['java', '-cp', lib_path, 'io.joern.joerncli.JoernParse']
    
    cpg_bin = os.path.join(tmpdir, 'test.bin')
    cmd = parse_cmd + [test_py, '-o', cpg_bin]
    
    print(f"\nRunning: {' '.join(cmd[:3])} ... [rest omitted]")
    print(f"Parsing to: {cpg_bin}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(f"\nReturn code: {result.returncode}")
        print(f"STDOUT: {result.stdout[:500] if result.stdout else '(empty)'}")
        print(f"STDERR: {result.stderr[:500] if result.stderr else '(empty)'}")
        
        if os.path.exists(cpg_bin):
            size = os.path.getsize(cpg_bin)
            print(f"✓ CPG binary created: {size} bytes")
        else:
            print("✗ CPG binary NOT created!")
            
    except subprocess.TimeoutExpired:
        print("ERROR: Command timed out (>30s)")
    except Exception as e:
        print(f"ERROR: {e}")

# Test Joern export command
print("\n" + "=" * 60)
print("3. Testing Joern Export...")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    test_py = os.path.join(tmpdir, 'test.py')
    with open(test_py, 'w') as f:
        f.write('def hello():\n    print("hello")\n')
    
    cpg_bin = os.path.join(tmpdir, 'test.bin')
    out_dir = os.path.join(tmpdir, 'out')
    # Don't create out_dir - Joern expects it NOT to exist
    
    # First parse
    lib_path = os.path.join(JOERN_PATH, "lib", "*")
    parse_cmd = ['java', '-cp', lib_path, 'io.joern.joerncli.JoernParse']
    
    print(f"Step 1: Parsing...")
    result = subprocess.run(
        parse_cmd + [test_py, '-o', cpg_bin],
        capture_output=True, text=True, timeout=30
    )
    
    if result.returncode != 0:
        print(f"Parse failed: {result.stderr[:200]}")
    else:
        print(f"✓ Parse successful (return code: 0)")
        
        # Then export
        print(f"\nStep 2: Exporting...")
        export_cmd = ['java', '-cp', lib_path, 'io.joern.joerncli.JoernExport']
        cmd = export_cmd + [cpg_bin, '-o', out_dir, '--repr', 'all', '--format', 'graphml']
        
        print(f"Running: {' '.join(cmd[:3])} ...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        print(f"\nReturn code: {result.returncode}")
        print(f"STDOUT: {result.stdout[:500] if result.stdout else '(empty)'}")
        print(f"STDERR: {result.stderr[:500] if result.stderr else '(empty)'}")
        
        graphml_path = os.path.join(out_dir, 'export.xml')
        if os.path.exists(graphml_path):
            size = os.path.getsize(graphml_path)
            print(f"✓ GraphML created: {size} bytes")
            with open(graphml_path) as f:
                first_lines = f.read(200)
                print(f"First 200 chars: {first_lines}")
        else:
            print(f"✗ GraphML NOT created at {graphml_path}")
            print(f"Files in {out_dir}:")
            for f in os.listdir(out_dir):
                print(f"  - {f}")

print("\n" + "=" * 60)
print("Debug complete!")
print("=" * 60)
