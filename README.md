# ASL Fingerspelling Data Processing Pipeline

This repository contains the data processing pipeline for the American Sign Language (ASL) Fingerspelling Kaggle competition.

The core script, `process_asl_data.py`, is designed to efficiently extract relevant 3D hand coordinates from massive parquet files, filter for conversational greetings, and structure the data into padded NumPy arrays ready for Machine Learning models (like LSTMs or Transformers).

## 🚀 Features

- **Conversational Filtering:** Focuses the dataset on practical phrases by parsing `supplemental_metadata.csv` for greetings like "hello", "how", "name", etc.
- **Massive RAM Optimization:** Navigates Kaggle's strict 16GB memory limits.
  - Uses PyArrow column-filtering on disk instead of loading full `1629`-column Parquet dataframes into Pandas.
  - Excludes Face and Pose landmarks entirely to save space.
  - Implements aggressive end-of-loop garbage collection (`gc.collect`) to prevent memory bloating.
- **Sequence Normalization:** Filters Parquet data correctly using sequence IDs.
  - Handles MultiIndex lookup safety checks.
  - Excludes buggy sequences extending beyond 2,000 frames.
- **Machine Learning Output:** Dynamically pads varied-length sign sequences into fixed-length arrays (e.g. `(samples, max_frames, 63)`) and saves them as `.npy` tensors.

## 🛠️ Usage (Kaggle Environment)

1.  Copy the contents of `process_asl_data.py` into a new cell inside your Kaggle Notebook.
2.  Ensure you have added the Google - ASL Fingerspelling Dataset to your kernel.
3.  Run the script. It will automatically:
    - Scan `/kaggle/input/` for `supplemental_metadata.csv` and the matching `.parquet` files.
    - Print progress loops in 500-sequence increments.
    - Automatically output `X_data.npy` (your $(x, y, z)$ hand coordinate tensors) and `y_labels.npy` (the text phrases) to your Kaggle Output directory.

## 📊 Next Steps After Processing

Once the script generates `X_data.npy` and `y_labels.npy`, you can load them directly into your Neural Network architecture in a new cell:

```python
import numpy as np

# Load the structured dataset
X = np.load('X_data.npy')
y = np.load('y_labels.npy')

print(f"X shape: {X.shape}") # Should look like (5186, <max_frames>, 63)
print(f"Y shape: {y.shape}") # Should look like (5186,)
```
