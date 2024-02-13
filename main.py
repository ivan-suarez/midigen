# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

!pip install pretty_midi

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

import numpy as np

# Example: one-hot encoding pitches
n_vocab = 128  # Number of possible pitches in MIDI
encoded_sequences = np.zeros((len(sequences), sequence_length, n_vocab))

for i, sequence in enumerate(sequences):
    for j, note in enumerate(sequence):
        pitch = note[2]  # Assuming the third element is the pitch
        encoded_sequences[i, j, pitch] = 1

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(encoded_sequences, next_notes, test_size=0.2)

np.save('/X_train.npy', X_train)
np.save('/X_val.npy', X_val)
# Save y_train and y_val similarly

