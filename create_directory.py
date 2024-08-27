import os
import numpy as np
from config import actions,DATA_PATH,no_sequences

# Create MP_Data directory if it doesn't exist
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
# Create folders for each action and sequence
for action in actions:
    action_path = os.path.join(DATA_PATH, action)

    # Create the action folder if it doesn't exist
    if not os.path.exists(action_path):
        os.makedirs(action_path)

    # Get the max directory number, or start from 0 if the directory is empty
    try:
        dirmax = np.max(np.array(os.listdir(action_path)).astype(int))
    except ValueError:  # if the folder is empty
        dirmax = -1  # Start from 0

    # Create subdirectories for each sequence
    for sequence in range(1, no_sequences + 1):
        try:
            os.makedirs(os.path.join(action_path, str(dirmax + sequence)))
        except Exception as e:
            print(f"An error occurred while creating directory: {e}")