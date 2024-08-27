import os
import signal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# Import the necessary modules
from config import actions
from preprocces import X_train, X_test, y_train, y_test

# Set up log directory for TensorBoard
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Set up checkpoint to save the model at regular intervals
checkpoint_dir = 'checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Use a file path with a unique name to avoid overwriting
checkpoint_filepath = os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.keras')
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,  # Save the full model (architecture + weights)
    save_best_only=False,     # Save the model at the end of every epoch
    save_freq='epoch',        # Save after every epoch
    verbose=1
)

# Define the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Define a handler to save the model when interrupted
def save_model_on_interrupt(signum, frame):
    print("\nTraining interrupted. Saving the model...")
    model.save('interrupted_model.keras')
    print("Model saved successfully!")
    exit(0)

# Attach the handler to the interrupt signal (Ctrl+C)
signal.signal(signal.SIGINT, save_model_on_interrupt)

# Train the model with the checkpoint callback
model.fit(X_train, y_train, epochs=2000, validation_data=(X_test, y_test),
          callbacks=[tb_callback, checkpoint_callback])

# Save the final model
model.save('final_model.keras')
