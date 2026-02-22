import numpy as np
import pandas as pd
import pickle
from datasets import load_dataset
from tqdm import tqdm


def code_to_image(code: str, image_size: int = 32) -> np.ndarray:
    """
    Below is a common mapping we have used for converting code characters to pixel values in the resulting image. This mapping is based on ASCII values and common code formatting:
        Pixel Value | What It Represents
        ─────────────────────────────────
        0       | Padding (unused, short code)
        10       | Newline character (\n)
        32       | Space character
        40-57    | Parentheses, numbers, punctuation
        65-90    | Uppercase letters (A-Z)
        97-122   | Lowercase letters (a-z)
        255      | High Unicode characters (rare in ASCII code)
    """
    code_bytes = code.encode('utf-8', errors='ignore')
    total_pixels = image_size * image_size
    if len(code_bytes) >= total_pixels:
        pixels = list(code_bytes[:total_pixels])
    else:
        pixels = list(code_bytes) + [0] * (total_pixels - len(code_bytes))
    return np.array(pixels, dtype=np.uint8).reshape(image_size, image_size)


def load_and_convert_dataset(output_path: str = 'dataset/code_images.parquet') -> pd.DataFrame:
    print("Loading HuggingFace dataset...")
    dataset = load_dataset("DaniilOr/CoDET-M4", split='train')

    print(f"Dataset loaded: {len(dataset)} samples")
    dataset = dataset.filter(lambda x: x['split'] == 'val')
    python_samples = dataset.filter(lambda x: x['language'] == 'python')
    print(f"Python samples: {len(python_samples)}")

    data_records = []
    print("Converting code to images...")
    for sample in tqdm(python_samples):
        if 'code' in sample and sample['code']:
            image_array = code_to_image(sample['code'])
            image_bytes = pickle.dumps(image_array)

            record = {
                'code_text': sample['code'],
                'code_image': image_bytes,
                'model': sample.get('model'),
                'target': sample.get('target'),
                'language': sample.get('language'),
            }

            for key in sample.keys():
                if key not in record and key not in ['code']:
                    record[key] = sample.get(key)

            data_records.append(record)

    print(f"Converted {len(data_records)} code samples to images")

    df = pd.DataFrame(data_records)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")

    df.to_parquet(output_path)
    print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    df = load_and_convert_dataset()
    print("\nDataFrame info:")
    print(df.info())
    if len(df) > 0:
        image = pickle.loads(df['code_image'].iloc[0])
        print(f"First row code image shape: {image.shape}")
