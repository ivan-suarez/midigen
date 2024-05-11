import pretty_midi
import os
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from keras.utils import to_categorical

midi_files = [f for f in os.listdir('data/') if f.endswith('.mid')]

midis = []
for file in midi_files:
    try:
        midi = pretty_midi.PrettyMIDI('data/' + file)
        midis.append(midi)
    except Exception as e:
        print(f"Error loading {file}: {e}")


#Parse MIDI Data
notes = []
for midi in midis:
    for instrument in midi.instruments:
        for note in instrument.notes:
            note_info = note.pitch#(note.start, note.end, note.pitch, note.velocity)
            notes.append(note_info)



#quantized_notes = quantize_notes(notes)
def process_and_encode_notes(midis):
    all_notes = []
    for midi in midis:
        for instrument in midi.instruments:
            for note in instrument.notes:
                all_notes.append(note.pitch)
    return all_notes

notes = process_and_encode_notes(midis)

num_classes = 128  # Number of MIDI note pitches
encoded_notes = to_categorical(notes, num_classes=num_classes)

sequence_length = 10  # Number of notes in a sequence
sequences = []
next_notes = []

for i in range(len(encoded_notes) - sequence_length):
    sequences.append(encoded_notes[i:i + sequence_length])
    next_notes.append(encoded_notes[i + sequence_length])

sequences = np.array(sequences)
next_notes = np.array(next_notes)



from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(sequence_length, num_classes)),
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
# Fit the model to the training data
history = model.fit(sequences, next_notes,
                epochs=50,  # Number of epochs to train for
                batch_size=64,  # Size of the batches of data
                verbose=1)  # Show training log

model.save('models/LSTM_0.6.2.keras')