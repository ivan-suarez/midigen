import pretty_midi
import os

midi_files = [f for f in os.listdir('clean/') if f.endswith('.mid')]

midis = []
for file in midi_files:
    try:
        midi = pretty_midi.PrettyMIDI('clean/' + file)
        midis.append(midi)
    except Exception as e:
        print(f"Error loading {file}: {e}")


#Parse MIDI Data
notes = []
for midi in midis:
    for instrument in midi.instruments:
        for note in instrument.notes:
            note_info = (note.start, note.end, note.pitch, note.velocity)
            notes.append(note_info)


def process_notes(midis):
    all_notes = []
    for midi in midis:
        for instrument in midi.instruments:
            prev_end_time = 0
            for note in instrument.notes:
                relative_start = note.start - prev_end_time
                duration = note.end - note.start  # Calculate duration
                note_info = (relative_start, duration, note.pitch, note.velocity)
                all_notes.append(note_info)
                prev_end_time = note.end
    return all_notes


def quantize_notes(notes, time_step=0.25):
    # Quantize start times and durations to the nearest time_step
    quantized_notes = []
    for note in notes:
        start, end, pitch, velocity = note
        quantized_start = round(start / time_step) * time_step
        quantized_end = round(end / time_step) * time_step
        quantized_notes.append((quantized_start, quantized_end, pitch, velocity))
    return quantized_notes

#quantized_notes = quantize_notes(notes)

quantized_notes = process_notes(midis)

sequence_length = 10  # Number of notes in a sequence
sequences = []
next_notes = []

#for i in range(0, len(quantized_notes) - sequence_length):
#    sequences.append(quantized_notes[i:i + sequence_length])
#    next_notes.append(quantized_notes[i + sequence_length])

for i in range(0, len(quantized_notes) - sequence_length):
    sequences.append(quantized_notes[i:i + sequence_length])
    next_notes.append(quantized_notes[i + sequence_length])

#for i in range(0, len(quantized_notes) - sequence_length):
#    sequences.append(quantized_notes[i:i +sequence_length])
#    start_time, end_time, pitch, velocity = quantized_notes[i + sequence_length]
 #   duration = end_time - start_time
 #   next_notes.append((start_time, duration, pitch, velocity))

# You may need to further process these sequences depending on your network architecture


# Example: one-hot encoding pitches
#n_vocab = 128  # Number of possible pitches in MIDI
#encoded_sequences = np.zeros((len(sequences), sequence_length, n_vocab))

#for i, sequence in enumerate(sequences):
#    for j, note in enumerate(sequence):
#        pitch = note[2]  # Assuming the third element is the pitch
#        encoded_sequences[i, j, pitch] = 1

#from sklearn.model_selection import train_test_split

#X_train, X_val, y_train, y_val = train_test_split(encoded_sequences, next_notes, test_size=0.2)

#np.save('/X_train.npy', X_train)
#np.save('/X_val.npy', X_val)
# Save y_train and y_val similarly

# Assuming you have already defined and compiled your model as 'model'

from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(sequence_length, 4)),  # Adjust the number of units based on model complexity
    LSTM(128),  # You can add more LSTM layers or adjust units
    Dense(64, activation='relu'),  # Intermediate dense layer, optional
    Dense(4)  # Output layer with 4 units for the four elements of the output tuple
])

model.compile(optimizer='adam', loss='mean_squared_error')

print(model.summary())
# Fit the model to the training data
history = model.fit(sequences, next_notes,
                epochs=50,  # Number of epochs to train for
                batch_size=64,  # Size of the batches of data
                verbose=1)  # Show training log

model.save('models/LSTM_0.3.1.keras')
