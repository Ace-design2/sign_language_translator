import os
import pandas as pd
import numpy as np
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
    # Add sequence_id to loaded columns so we can filter by it later
    cols_to_load = ['sequence_id', 'frame'] + get_hand_columns()
    
    try:
        table = pq.read_table(parquet_path, columns=cols_to_load)
    except ValueError as e:
        print(f"Default columns not found, attempting alternative schema. ({e})")
        cols_to_load = ['sequence_id', 'frame']
        for hand in ['left_hand', 'right_hand']:
            for i in range(21):
                for dim in ['x', 'y', 'z']:
                    cols_to_load.append(f'{hand}_{i}_{dim}')
        
        # If sequence_id isn't a top-level column, it might be the row index
        # PyArrow allows loading without it and we'll check the index
        try:
            table = pq.read_table(parquet_path, columns=cols_to_load)
        except ValueError:
            # Fallback stringently without sequence_id in cols_to_load, relying on index
            cols_to_load = [c for c in cols_to_load if c != 'sequence_id']
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

    if not metadata_path:
        print("❌ Missing metadata file.")
        return

    df = pd.read_csv(metadata_path)
    
    # 2. Update filter to find exact words (using \b for word boundaries so 'this' doesn't trigger 'hi')
    keywords = [r'\bhow\b', r'\byou\b', r'\bmeet\b', r'\bname\b', r'\bhello\b', r'\bhi\b', r'\bthank\b', r'\bwant\b']
    greetings_df = df[df['phrase'].astype(str).str.contains('|'.join(keywords), case=False, na=False, regex=True)]

    print(f"Filtered to {len(greetings_df)} relevant introduction/greeting clips.")
    
    # --- THIS IS THE NEW PART: BUILDING THE DATASET ---
    
    X_data = []
    y_labels = []
    
    # Process ALL filtered greetings instead of just 10!
    # Note: 5186 clips will take a bit of time to process on Kaggle, so we add a counter.
    total_clips = len(greetings_df)
    
    for idx, row in greetings_df.iterrows():
        phrase = row['phrase']
        sample_path = row['path']
        sequence_id = row['sequence_id']
        full_parquet_path = find_file(os.path.basename(sample_path))

        if full_parquet_path:
            landmarks_df = load_hand_landmarks(full_parquet_path)
            
            # 1. Filter the huge dataframe down to JUST this sequence's video frames
            if 'sequence_id' in landmarks_df.columns:
                landmarks_df = landmarks_df[landmarks_df['sequence_id'] == sequence_id]
            elif landmarks_df.index.name == 'sequence_id':
                try:
                    landmarks_df = landmarks_df.loc[sequence_id]
                except KeyError:
                    pass 
                
            # Get only right hand columns, ignore the 'frame' index for the math model
            hand_cols = [c for c in landmarks_df.columns if 'right_hand' in c]
            hand_data = landmarks_df[hand_cols].dropna() 
            
            # Convert the Pandas DataFrame into a raw Math Array (NumPy)
            sequence_array = hand_data.values 
            
            if len(sequence_array) > 0:
                X_data.append(sequence_array)
                y_labels.append(phrase)
                
        # --- Print progress every 500 videos so Kaggle doesn't look frozen! ---
        if len(X_data) % 500 == 0 and len(X_data) > 0:
            print(f"⏳ Processed {len(X_data)} / {total_clips} conversational sequences...")
            print(f"✅ Last extracted array for: '{phrase}' -> Shape: {sequence_array.shape}")

    print("\n--- DATA EXTRACTION COMPLETE ---")
    print(f"Successfully built {len(X_data)} sequences of X (features) and Y (labels)")
    
    if len(X_data) > 0:
        # --- NEW: PADDING SEQUENCES ---
        # Neural networks need all inputs to be the exact same shape!
        # Find the longest video sequence in our batch
        max_length = max([seq.shape[0] for seq in X_data])
        print(f"Padding all sequences to {max_length} frames...")
        
        # Pad all shorter sequences with zeroes to match the longest one
        X_padded = []
        for seq in X_data:
            # How many blank frames do we need to add to the end?
            pad_amount = max_length - seq.shape[0]
            # Pad with 0s at the bottom (axis 0), do nothing to the 63 coordinates (axis 1)
            padded_seq = np.pad(seq, ((0, pad_amount), (0, 0)), mode='constant', constant_values=0)
            X_padded.append(padded_seq)
            
        # Convert lists to final Machine Learning Tensors (Numpy Arrays)
        X = np.array(X_padded)
        y = np.array(y_labels)
        
        print("\n--- READY FOR MACHINE LEARNING ---")
        print(f"Final X Shape (Videos, Frames, Coordinates): {X.shape}") 
        print(f"Final y Shape (Labels): {y.shape}")
        
        # Save them to disk so you can load them directly into TensorFlow later
        np.save("X_data.npy", X)
        np.save("y_labels.npy", y)
        print("✅ Saved 'X_data.npy' and 'y_labels.npy' to Kaggle Output folder!")

if __name__ == "__main__":
    main()
