from datasets import load_dataset

print("Loading dataset...")
dataset = load_dataset("DaniilOr/CoDET-M4", split='train')
dataset = dataset.filter(lambda x: x['split'] == 'val')
python_samples = dataset.filter(lambda x: x['language'] == 'python')

print(f"Total Python samples: {len(python_samples)}")
print(f"Columns: {python_samples.column_names}")
print(f"\nFirst sample:")
sample = python_samples[0]
for key in python_samples.column_names:
    val = sample[key]
    if isinstance(val, str):
        if len(val) > 100:
            print(f"  {key}: {val[:100]}...")
        else:
            print(f"  {key}: {val}")
    else:
        print(f"  {key}: {type(val).__name__}")
