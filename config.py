import os
import numpy as np

# Assuming aclstm folder is on your Desktop
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'pylstm')

# Path for exported data, numpy arrays
DATA_PATH = os.path.join(desktop_path, 'MP_Data')

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Thirty videos worth of data for each sign
no_sequences = 30

# Videos are going to be 30 frames each in length
sequence_length = 30

# Folder start
start_folder = 0

