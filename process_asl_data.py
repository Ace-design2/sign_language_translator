import os
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure environment - Easily switch between Kaggle and local environments
IS_KAGGLE = True  # Make sure this is True when on the website

if IS_KAGGLE:
    # This is the standard path for this specific Google dataset on Kaggle
    BASE_DIR = '/kaggle/input/asl-fingerspelling'
else:
    BASE_DIR = './data'

train_csv_path = os.path.join(BASE_DIR, 'train.csv')

if os.path.exists(train_csv_path):
    df = pd.read_csv(train_csv_path)
    print(f"Success! Loaded {len(df)} rows.")
else:
    print(f"Error: '{train_csv_path}' not found.")
    # This will list all files available so you can see the correct folder name
    print("Available files:", os.listdir('/kaggle/input/'))

def get_hand_columns():
    """
    Generates column names for the 21 hand landmarks (x, y, z) for both hands.
    These match the expected pyarrow dataframe columns to load only what is needed.
    """
    cols = []
    # Both x_left_hand_0 and left_hand_0_x conventions can occur; we'll define standard Kaggle ASL ones
    for hand in ['left_hand', 'right_hand']:
        for dim in ['x', 'y', 'z']:
            for i in range(21):
                cols.append(f'{dim}_{hand}_{i}')
    return cols

def filter_intro_phrases(csv_path):
    """
    Loads train.csv and filters for phrases containing introduction keywords
    like "name", "hello", etc.
    """
    print(f"Loading '{csv_path}' to filter phrases...")
    df = pd.read_csv(csv_path)
    
    # Common introduction keywords or common name-spelling sequences
    keywords = ['name', 'hello', 'hi ', 'meet']
    pattern = '|'.join(keywords)
    
    # Filter where phrase matches the pattern (case insensitive)
    filtered_df = df[df['phrase'].astype(str).str.contains(pattern, case=False, na=False)]
    return filtered_df

def load_hand_landmarks(parquet_path):
    """
    Loads a single parquet file, extracting ONLY hand landmarks to save memory.
    This ignores face and pose landmarks entirely.
    """
    print(f"Loading hand landmarks from {parquet_path}...")
    
    # Often Kaggle ASL datasets have `frame` or `sequence_id` as index/columns.
    cols_to_load = ['frame'] + get_hand_columns()
    
    try:
        # Load the requested columns directly via PyArrow to conserve RAM
        table = pq.read_table(parquet_path, columns=cols_to_load)
    except ValueError as e:
        # Handle case where column names might be slightly varied like `left_hand_0_x`
        print(f"Default columns not found, attempting alternative schema. ({e})")
        cols_to_load = ['frame']
        for hand in ['left_hand', 'right_hand']:
            for i in range(21):
                for dim in ['x', 'y', 'z']:
                    cols_to_load.append(f'{hand}_{i}_{dim}')
        table = pq.read_table(parquet_path, columns=cols_to_load)

    # Convert to Pandas DataFrame
    df = table.to_pandas()
    return df

def plot_first_valid_frame(df):
    """
    Plots the 21 landmarks of the left or right hand for the first frame 
    that contains valid (non-NaN) hand data.
    """
    # Look for the exact format of the loaded columns
    cols = df.columns.tolist()
    right_x_col = next((c for c in cols if 'right_hand_0' in c and ('x_' in c or '_x' in c)), None)
    left_x_col = next((c for c in cols if 'left_hand_0' in c and ('x_' in c or '_x' in c)), None)
    
    # Attempt right hand first, fallback to left
    if right_x_col and df[right_x_col].notna().any():
        hand_type = 'right_hand'
        valid_frames = df[df[right_x_col].notna()]
    elif left_x_col and df[left_x_col].notna().any():
        hand_type = 'left_hand'
        valid_frames = df[df[left_x_col].notna()]
    else:
        print("No valid hand landmarks (left or right) found in this sequence.")
        return

    # Extract the very first valid frame
    first_frame = valid_frames.iloc[0]
    frame_idx = first_frame['frame'] if 'frame' in first_frame else 0
    print(f"Plotting {hand_type} at frame {frame_idx}")
    
    # Gather x, y, z arrays
    xs, ys, zs = [], [], []
    for i in range(21):
        # Accommodate both `x_right_hand_0` and `right_hand_0_x` schema styles
        x_val = first_frame.get(f'x_{hand_type}_{i}', first_frame.get(f'{hand_type}_{i}_x'))
        y_val = first_frame.get(f'y_{hand_type}_{i}', first_frame.get(f'{hand_type}_{i}_y'))
        z_val = first_frame.get(f'z_{hand_type}_{i}', first_frame.get(f'{hand_type}_{i}_z'))
        xs.append(x_val)
        ys.append(y_val)
        zs.append(z_val)
        
    # Visualizing the 3D hand coordinates
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c='b', marker='o')
    
    # Map the standard 21 MediaPipe hand connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),                     # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),                     # Index
        (5, 9), (9, 10), (10, 11), (11, 12),                # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),              # Ring
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)     # Pinky
    ]
    
    for (i, j) in connections:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], c='r')
        
    ax.set_title(f"Hand Landmarks ({hand_type}) - Frame {frame_idx}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Invert the Y and Z axis so the hand appears upright (as interpreted by MediaPipe)
    ax.invert_yaxis()
    ax.invert_zaxis()
    
    plt.show()

def main():
    print("="*50)
    print(f"Running with IS_KAGGLE = {IS_KAGGLE}")
    print(f"Looking for dataset in: {BASE_DIR}")
    print("="*50)
    
    if not os.path.exists(train_csv_path):
        print(f"Error: '{train_csv_path}' not found.")
        print("Please place your local data in the expected directory, or flip IS_KAGGLE to True if running on Kaggle.")
        return
        
    # 1. Filter train.csv
    filtered_df = filter_intro_phrases(train_csv_path)
    print(f"Found {len(filtered_df)} phrases matching introduction keywords.")
    
    if len(filtered_df) > 0:
        # 2. Get the file path of the first match
        first_file_rel_path = filtered_df.iloc[0]['path']
        full_parquet_path = os.path.join(BASE_DIR, first_file_rel_path)
        
        if os.path.exists(full_parquet_path):
            # 3. Load single parquet utilizing PyArrow features for low memory
            landmarks_df = load_hand_landmarks(full_parquet_path)
            
            # 4. Filter for single frame and plot 21 nodes
            print("Plotting the first valid frame...")
            plot_first_valid_frame(landmarks_df)
            
        else:
            print(f"Parquet file not found: {full_parquet_path}")

if __name__ == "__main__":
    main()
