import os
import numpy as np
import pretty_midi
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, LSTM, Dense, GlobalAveragePooling1D, Dropout, Attention, LayerNormalization
import keras

# 1. Load MIDI files from the 'data/classical' directory
midi_files = [os.path.join(root, file)
              for root, dirs, files in os.walk('data/classical')
              for file in files if file.endswith('.mid')]

midis = []
for file in midi_files:
    try:
        midi = pretty_midi.PrettyMIDI(file)
        midis.append(midi)
    except Exception as e:
        print(f"Error loading {file}: {e}")

# 2. Quantize note durations using fixed bins
def quantize_duration(note_duration, tempo):
    # Calculate duration relative to a quarter note
    quarter_note_duration = 60 / tempo
    duration_ratio = note_duration / quarter_note_duration
    
    # Define bin edges for duration categories
    # Category 0: <= 0.125 (32nd notes and shorter)
    # Category 1: > 0.125 and <= 0.25 (16th notes)
    # Category 2: > 0.25 and <= 0.5 (8th notes)
    # Category 3: > 0.5 and <= 0.75 (dotted 8th notes)
    # Category 4: > 0.75 and <= 1.0 (quarter notes)
    # Category 5: > 1.0 and <= 1.5 (dotted quarter notes)
    # Category 6: > 1.5 and <= 2.0 (half notes)
    # Category 7: > 2.0 and <= 3.0 (dotted half notes)
    # Category 8: > 3.0 (whole notes and longer)
    bins = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf]
    
    # Debug info (using a static counter to limit output)
    if not hasattr(quantize_duration, 'debug_count'):
        quantize_duration.debug_count = 0
    
    if quantize_duration.debug_count < 5:
        print(f"Note duration: {note_duration:.4f}s, Quarter note: {quarter_note_duration:.4f}s")
        print(f"Duration ratio: {duration_ratio:.4f}, Category: {np.digitize(duration_ratio, bins)}")
        quantize_duration.debug_count += 1
    
    category_index = np.digitize(duration_ratio, bins)
    return category_index

# 3. Process MIDI files to extract note pitches and quantized durations
def process_and_encode_notes(midis):
    notes = []
    durations = []
    for midi in midis:
        # Use the first tempo value or default to 120 BPM if not available
        try:
            tempo_changes = midi.get_tempo_changes()
            tempo = tempo_changes[1][0] if len(tempo_changes[1]) > 0 else 120
        except Exception:
            tempo = 120
        for instrument in midi.instruments:
            # Optionally skip percussion instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    note_pitch = note.pitch
                    note_duration = note.end - note.start
                    quantized_duration = quantize_duration(note_duration, tempo)
                    notes.append(note_pitch)
                    durations.append(quantized_duration)
    return notes, durations

notes, durations = process_and_encode_notes(midis)
print(f"First 10 notes: {notes[:10]}")
print(f"First 10 durations: {durations[:10]}")

# 4. Define number of classes for pitch and duration
num_pitch_classes = 128  # MIDI pitches 0-127
num_duration_classes = 9  # np.digitize with bins produces values 0..8

# 5. Prepare sequences for training
sequence_length = 10
input_sequences = []
target_pitches = []
target_durations = []

# Create sequences where each input is a sequence of concatenated one-hot vectors,
# and targets are the next note's pitch and duration (one-hot encoded separately)
for i in range(len(notes) - sequence_length):
    seq = []
    for j in range(i, i + sequence_length):
        pitch_vec = to_categorical(notes[j], num_classes=num_pitch_classes)
        duration_vec = to_categorical(durations[j], num_classes=num_duration_classes)
        combined = np.concatenate((pitch_vec, duration_vec))
        seq.append(combined)
    input_sequences.append(seq)
    
    target_pitch = to_categorical(notes[i + sequence_length], num_classes=num_pitch_classes)
    target_duration = to_categorical(durations[i + sequence_length], num_classes=num_duration_classes)
    target_pitches.append(target_pitch)
    target_durations.append(target_duration)

input_sequences = np.array(input_sequences)
target_pitches = np.array(target_pitches)
target_durations = np.array(target_durations)

print("Input Sequences Shape:", input_sequences.shape)
print("Target Pitches Shape:", target_pitches.shape)
print("Target Durations Shape:", target_durations.shape)

# 6. Build a multi-output LSTM model using the Functional API
input_shape = (sequence_length, num_pitch_classes + num_duration_classes)
inputs = Input(shape=input_shape)

# First LSTM layer
lstm1 = LSTM(256, return_sequences=True, recurrent_dropout=0.1)(inputs)
lstm1 = LayerNormalization()(lstm1)

# Second LSTM layer
lstm2 = LSTM(128, return_sequences=True, recurrent_dropout=0.1)(lstm1)
lstm2 = LayerNormalization()(lstm2)

# Apply Keras built-in Attention layer
# Self-attention on the sequence
attention_output = Attention()([lstm2, lstm2])
# Add dropout for regularization
attention_output = Dropout(0.2)(attention_output)

# Final LSTM layer processes the attention-weighted sequence
lstm3 = LSTM(128)(attention_output)
lstm3 = Dropout(0.2)(lstm3)

# Dense layer for shared representation
shared = Dense(128, activation='relu')(lstm3)

# Separate output heads for pitch and duration
pitch_output = Dense(num_pitch_classes, activation='softmax', name='pitch_output')(shared)
duration_output = Dense(num_duration_classes, activation='softmax', name='duration_output')(shared)

model = Model(inputs=inputs, outputs=[pitch_output, duration_output])
model.compile(optimizer='adam',
              loss={'pitch_output': 'categorical_crossentropy',
                    'duration_output': 'categorical_crossentropy'},
              metrics=['accuracy'])

print(model.summary())

# 7. Train the model
history = model.fit(input_sequences,
                    {'pitch_output': target_pitches, 'duration_output': target_durations},
                    epochs=50,
                    batch_size=64,
                    verbose=1)

# 8. Save the model (create directory if it doesn't exist)
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(os.path.join(model_dir, 'LSTM_multi_output.keras'))
