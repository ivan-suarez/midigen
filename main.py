import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import pretty_midi
import os

midi_files = [f for f in os.listdir('/kaggle/input/classical-music-midi/bach/') if f.endswith('.mid')]

midis = []
for file in midi_files:
    try:
        midi = pretty_midi.PrettyMIDI('/kaggle/input/classical-music-midi/bach/' + file)
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

def quantize_notes(notes, time_step=0.25):
    # Quantize start times and durations to the nearest time_step
    quantized_notes = []
    for note in notes:
        start, end, pitch, velocity = note
        quantized_start = round(start / time_step) * time_step
        quantized_end = round(end / time_step) * time_step
        quantized_notes.append((quantized_start, quantized_end, pitch, velocity))
    return quantized_notes

quantized_notes = quantize_notes(notes)

sequence_length = 50  # Number of notes in a sequence
sequences = []
next_notes = []

for i in range(0, len(quantized_notes) - sequence_length):
    sequences.append(quantized_notes[i:i + sequence_length])
    next_notes.append(quantized_notes[i + sequence_length])

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
    LSTM(256, return_sequences=True, input_shape=(50, 4)),  # Adjust the number of units based on model complexity
    LSTM(128),  # You can add more LSTM layers or adjust units
    Dense(64, activation='relu'),  # Intermediate dense layer, optional
    Dense(4)  # Output layer with 4 units for the four elements of the output tuple
])

model.compile(optimizer='adam', loss='mean_squared_error')
# Fit the model to the training data
history = model.fit(sequences, next_notes,
                    epochs=100,  # Number of epochs to train for
                    batch_size=64,  # Size of the batches of data
                    verbose=1)  # Show training log


def generate_notes(model, seed_sequence, num_notes_to_generate):
    generated_sequence = seed_sequence.copy()  # Copy the seed sequence
    for _ in range(num_notes_to_generate):
        # Reshape the sequence to match the model's input shape: (1, 50, 4)
        input_sequence = np.array(generated_sequence[-50:]).reshape(1, 50, 4)
        
        # Predict the next note
        predicted_note = model.predict(input_sequence)[0]  # model.predict returns a batch, so get the first
        
        # Append the predicted note to the sequence
        generated_sequence.append(predicted_note.tolist())
        
    return generated_sequence

# Assuming `x_train` is your training data and `seed_index` is some index of your choice
seed_index = 0
seed_sequence = sequences[seed_index][:50] # Starting with an existing sequence as seed
num_notes_to_generate = 100  # Number of notes you want to generate

# Generate notes
generated_notes = generate_notes(model, seed_sequence, num_notes_to_generate)


from mido import MidiFile, MidiTrack, Message

def create_midi_from_notes(notes, output_file='generated_sequence.mid'):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Assuming the first event happens at time 0
    last_event_time = 0
    
    for note in notes:
        start_time, end_time, pitch, velocity = map(int, note)
        
        # Ensure non-negative delta times
        delta_time_on = max(0, start_time - last_event_time)
        delta_time_off = max(0, end_time - start_time)
        
        # Update last event time for the next iteration
        # For note_on, this is the start time of the current note
        last_event_time = start_time
        
        # Add the note_on message
        track.append(Message('note_on', note=pitch, velocity=velocity, time=delta_time_on))
        # For note_off, update the last event time to be the end time of the current note
        last_event_time = end_time
        
        # Add the note_off message with time since the note_on
        track.append(Message('note_off', note=pitch, velocity=0, time=delta_time_off))
    
    midi.save(output_file)

output_path = '/kaggle/working/my_generated_midi.mid'
create_midi_from_notes(generated_notes, output_file=output_path)



