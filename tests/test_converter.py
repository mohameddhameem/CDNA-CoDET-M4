import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def code_to_image(code: str, image_size: int = 32) -> np.ndarray:
    code_bytes = code.encode('utf-8', errors='ignore')
    total_pixels = image_size * image_size
    if len(code_bytes) >= total_pixels:
        pixels = list(code_bytes[:total_pixels])
    else:
        pixels = list(code_bytes) + [0] * (total_pixels - len(code_bytes))
    image = np.array(pixels, dtype=np.uint8).reshape(image_size, image_size)
    return image


print("Loading dataset...")
dataset = load_dataset("DaniilOr/CoDET-M4", split='train')
print(f"Total: {len(dataset)}")

print("Filtering for val split...")
dataset = dataset.filter(lambda x: x['split'] == 'val')
print(f"Val: {len(dataset)}")

print("Filtering for python...")
python_samples = dataset.filter(lambda x: x['language'] == 'python')
print(f"Python: {len(python_samples)}")

print(f"Columns: {python_samples.column_names}")

data_records = []
print("Converting code to images...")
for idx, sample in enumerate(tqdm(python_samples)):
    if 'code' in sample and sample['code']:
        image_array = code_to_image(sample['code'])
        record = {
            'code_text': sample['code'],
            'code_image': image_array,
            'model': sample.get('model'),
            'target': sample.get('target'),
            'language': sample.get('language'),
        }
        for key in sample.keys():
            if key not in record and key not in ['code']:
                record[key] = sample.get(key)
        data_records.append(record)

print(f"Converted {len(data_records)} samples")

df = pd.DataFrame(data_records)
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

df.to_parquet('dataset/code_images.parquet')
print("Saved to dataset/code_images.parquet")

if len(df) > 0:
    print(f"First image shape: {df['code_image'].iloc[0].shape}")
    print(f"DataFrame info:\n{df.info()}")
