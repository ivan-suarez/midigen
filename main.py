import pretty_midi
import os
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from keras.utils import to_categorical

midi_files = [os.path.join(root, file) for root, dirs, files in os.walk('data/classical') for file in files if file.endswith('.mid')]

midis = []
for file in midi_files:
    try:
        midi = pretty_midi.PrettyMIDI(file)
        midis.append(midi)
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Function to quantize note durations
def quantize_duration(note_duration, tempo):
    # Convert note duration from seconds to a fraction of a whole note
    print(note_duration)
    quarter_note_duration = 60 / tempo  # Duration of a quarter note in seconds
    whole_note_duration = 4 * quarter_note_duration
    duration_ratio = note_duration / quarter_note_duration
    print(duration_ratio)
    # Define bins for whole, half, quarter, eighth, sixteenth notes (including a catch-all for longer durations)
    bins = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, np.inf]
    midpoints = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
    midpoints = [0] + midpoints + [np.inf]  # Add boundaries at 0 and inf for simplicity
    
    # Find the closest bin using midpoints
    category_index = np.digitize([duration_ratio], midpoints)[0] - 1
    print(bins[category_index])
    return bins[category_index]



#quantized_notes = quantize_notes(notes)
def process_and_encode_notes(midis):
    all_notes = []
    all_durations = []
    for midi in midis:
        tempo = midi.get_tempo_changes()[1][0]
        for instrument in midi.instruments:
            for note in instrument.notes:
                note_pitch = note.pitch
                note_duration = note.end - note.start
                quantized_duration = quantize_duration(note_duration, tempo)
                all_notes.append(note_pitch)
                all_durations.append(quantized_duration)
    return all_notes, all_durations

notes, durations = process_and_encode_notes(midis)



num_classes = 128  # Number of MIDI note pitches
num_duration_classes = 8
encoded_notes = to_categorical(notes, num_classes=num_classes)
encoded_durations = to_categorical(durations, num_classes=num_duration_classes)

encoded_features = np.concatenate((encoded_notes, encoded_durations), axis=1)


sequence_length = 10  # Number of notes in a sequence
sequences = []
next_features = []

for i in range(len(encoded_features) - sequence_length):
    sequences.append(encoded_features[i:i + sequence_length])
    next_features.append(encoded_features[i + sequence_length])

sequences = np.array(sequences)
next_features = np.array(next_features)



from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(sequence_length, num_classes + num_duration_classes)),#modify
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dense(num_classes + num_duration_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
# Fit the model to the training data
history = model.fit(sequences, next_features,
                epochs=50,  # Number of epochs to train for
                batch_size=64,  # Size of the batches of data
                verbose=1)  # Show training log

model.save('models/LSTM_0.7.1.keras')