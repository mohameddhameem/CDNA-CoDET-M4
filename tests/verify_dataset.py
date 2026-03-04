import pandas as pd
import numpy as np
import pickle

print("Loading parquet file...")
df = pd.read_parquet('dataset/code_images.parquet')

print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print(f"\nLoading and reconstructing first 3 images...")
for i in range(min(3, len(df))):
    image = pickle.loads(df['code_image'].iloc[i])
    print(f"  Image {i}: shape {image.shape}, dtype {image.dtype}, min={image.min()}, max={image.max()}")

print(f"\nDataset ready for CNN training!")
print(f"Total samples: {len(df)}")
print(f"Image size: 32x32")
print(f"File location: dataset/code_images.parquet")
