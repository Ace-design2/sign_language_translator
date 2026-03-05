import os
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Search for the conversational metadata file
def find_file(name, search_path='/kaggle/input/'):
    # Check current directory fallback for local testing
    if os.path.exists('./data/' + name):
        return os.path.join('./data/', name)
        
    for root, dirs, files in os.walk(search_path):
        if name in files:
            return os.path.join(root, name)
            
    # Fallback to recursively search current directory
    for root, dirs, files in os.walk('.'):
        if name in files:
            return os.path.join(root, name)
            
    return None

def get_hand_columns():
    """
    Generates column names for the 21 hand landmarks (x, y, z) for both hands.
    These match the expected pyarrow dataframe columns to load only what is needed directly from disk.
    """
    cols = []
    for hand in ['left_hand', 'right_hand']:
        for dim in ['x', 'y', 'z']:
            for i in range(21):
                cols.append(f'{dim}_{hand}_{i}')
    return cols

def load_hand_landmarks(parquet_path):
    """
    Loads a single parquet file, extracting ONLY hand landmarks to save memory.
    This ignores face and pose landmarks entirely.
    """
    # -------------------------------------------------------------------------
    # Note: We use PyArrow here instead of pd.read_parquet(parquet_path) 
    # because pd.read_parquet loads ALL 1629 columns into RAM first, 
    # before you can filter them. PyArrow filters them ON DISK during read, 
    # saving massive amounts of RAM on Kaggle.
    # -------------------------------------------------------------------------
    print(f"Loading hand landmarks from {parquet_path}...")
    cols_to_load = ['frame'] + get_hand_columns()
    
    try:
        table = pq.read_table(parquet_path, columns=cols_to_load)
    except ValueError as e:
        print(f"Default columns not found, attempting alternative schema. ({e})")
        cols_to_load = ['frame']
        for hand in ['left_hand', 'right_hand']:
            for i in range(21):
                for dim in ['x', 'y', 'z']:
                    cols_to_load.append(f'{hand}_{i}_{dim}')
        table = pq.read_table(parquet_path, columns=cols_to_load)

    return table.to_pandas()

def plot_first_valid_frame(df):
    """
    Plots the 21 landmarks of the left or right hand for the first frame 
    that contains valid (non-NaN) hand data.
    """
    cols = df.columns.tolist()
    right_x_col = next((c for c in cols if 'right_hand_0' in c and ('x_' in c or '_x' in c)), None)
    left_x_col = next((c for c in cols if 'left_hand_0' in c and ('x_' in c or '_x' in c)), None)
    
    if right_x_col and df[right_x_col].notna().any():
        hand_type = 'right_hand'
        valid_frames = df[df[right_x_col].notna()]
    elif left_x_col and df[left_x_col].notna().any():
        hand_type = 'left_hand'
        valid_frames = df[df[left_x_col].notna()]
    else:
        print("No valid hand landmarks (left or right) found in this sequence.")
        return

    first_frame = valid_frames.iloc[0]
    frame_idx = first_frame['frame'] if 'frame' in first_frame else 0
    print(f"Plotting {hand_type} at frame {frame_idx}")
    
    xs, ys, zs = [], [], []
    for i in range(21):
        x_val = first_frame.get(f'x_{hand_type}_{i}', first_frame.get(f'{hand_type}_{i}_x'))
        y_val = first_frame.get(f'y_{hand_type}_{i}', first_frame.get(f'{hand_type}_{i}_y'))
        z_val = first_frame.get(f'z_{hand_type}_{i}', first_frame.get(f'{hand_type}_{i}_z'))
        xs.append(x_val)
        ys.append(y_val)
        zs.append(z_val)
        
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c='b', marker='o')
    
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),                     
        (0, 5), (5, 6), (6, 7), (7, 8),                     
        (5, 9), (9, 10), (10, 11), (11, 12),                
        (9, 13), (13, 14), (14, 15), (15, 16),              
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)     
    ]
    for (i, j) in connections:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], c='r')
        
    ax.set_title(f"Hand Landmarks ({hand_type}) - Frame {frame_idx}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.invert_yaxis()
    ax.invert_zaxis()
    plt.show()

def main():
    print("="*50)
    print("Searching for conversational metadata...")
    print("="*50)
    
    # 1. Use supplemental_metadata for sentences/greetings!
    metadata_path = find_file('supplemental_metadata.csv')

    if metadata_path:
        df = pd.read_csv(metadata_path)
        print(f"✅ Found conversational data at: {metadata_path}")
        print(f"Loaded {len(df)} sentence sequences.")
    else:
        print("❌ Still missing. Check the 'Data' tab in your Kaggle sidebar and ensure the dataset is added.")
        return

    # 2. Filter for basic greeting/intro keywords
    keywords = ['how', 'you', 'meet', 'name', 'hello', 'hi', 'thank', 'want']
    greetings_df = df[df['phrase'].astype(str).str.contains('|'.join(keywords), case=False, na=False)]

    print(f"Filtered to {len(greetings_df)} relevant introduction/greeting clips.")
    if len(greetings_df) > 0:
        print(greetings_df[['phrase', 'path']].head())

        # 3. Get the path for the first greeting in your list
        sample_path = greetings_df.iloc[0]['path']
        
        # Kaggle paths in the CSV are relative, so we fix them:
        full_parquet_path = find_file(os.path.basename(sample_path))

        if full_parquet_path:
            # Load only the hands to save 180GB of headaches
            # Each parquet file contains 1629 columns; we only need a few.
            landmarks_df = load_hand_landmarks(full_parquet_path)
            
            # Filter for just the Right Hand coordinates (21 points * 3 axes = 63 columns)
            hand_cols = [c for c in landmarks_df.columns if 'right_hand' in c]
            if 'frame' in landmarks_df.columns:
                hand_cols = ['frame'] + hand_cols
                
            hand_data = landmarks_df[hand_cols].dropna() # Remove frames where hand wasn't found

            print(f"Extracted {len(hand_data)} frames of right-hand movement for the phrase: '{greetings_df.iloc[0]['phrase']}'")
            
            print("Plotting the first valid frame...")
            plot_first_valid_frame(landmarks_df)
        else:
            print(f"❌ Parquet file not found for {os.path.basename(sample_path)}")

if __name__ == "__main__":
    main()
