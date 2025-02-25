from mido import MidiFile, MidiTrack, Message
from tensorflow.keras.models import load_model
import numpy as np
from keras.utils import to_categorical

def generate():
    # Load the multi-output model saved from training
    model = load_model('models/LSTM_multi_output.keras')
    
    num_pitches = 128  # MIDI pitches range 0-127
    num_duration_classes = 9  # As used in training (categories 0 through 8)
    
    # Define an example seed sequence of 10 notes
    seed_pitches = [56, 60, 63, 50, 47, 47, 56, 60, 62, 50]
    # For simplicity, assume a default duration category of 4 (e.g., quarter note) for each seed note
    seed_durations = to_categorical([4] * 10, num_classes=num_duration_classes)
    # One-hot encode seed pitches
    seed_pitch_ohe = to_categorical(seed_pitches, num_classes=num_pitches)
    # Concatenate pitch and duration features for each note (shape: [10, 128+9])
    seed_sequence = np.concatenate((seed_pitch_ohe, seed_durations), axis=1)
    
    num_notes_to_generate = 100
    generated_notes = generate_notes(model, seed_sequence, num_notes_to_generate, num_pitches, num_duration_classes)
    
    # Decode generated one-hot vectors into pitch and duration indices
    generated_pitches = [np.argmax(note[:num_pitches]) for note in generated_notes]
    generated_durations = [np.argmax(note[num_pitches:]) for note in generated_notes]
    
    create_midi_from_notes(generated_pitches, generated_durations)
    return generated_pitches, generated_durations

def generate_notes(model, seed_sequence, num_notes_to_generate, num_pitches, num_duration_classes):
    # Copy the seed sequence as a list of note feature vectors
    generated_sequence = seed_sequence.tolist()
    sequence_length = seed_sequence.shape[0]
    
    for _ in range(num_notes_to_generate):
        # Prepare the input: take the last 'sequence_length' notes and reshape for prediction
        input_seq = np.array(generated_sequence[-sequence_length:]).reshape(1, sequence_length, -1)
        # The model outputs two predictions: one for pitch and one for duration
        pitch_pred, duration_pred = model.predict(input_seq)
        # For the next note, take the argmax from each output (since we used softmax)
        pred_pitch_index = np.argmax(pitch_pred, axis=1)[0]
        pred_duration_index = np.argmax(duration_pred, axis=1)[0]
        # Convert the predicted indices back to one-hot vectors
        pitch_vec = to_categorical(pred_pitch_index, num_classes=num_pitches)
        duration_vec = to_categorical(pred_duration_index, num_classes=num_duration_classes)
        # Concatenate to form the full feature vector for the predicted note
        new_note = np.concatenate((pitch_vec, duration_vec))
        generated_sequence.append(new_note)
    
    # Return only the generated notes (exclude the initial seed)
    return generated_sequence[sequence_length:]

def create_midi_from_notes(pitches, durations, output_file='generated_sequence6.mid'):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Set a tempo (in BPM) and define ticks per beat (MIDI resolution)
    tempo = 120
    ticks_per_beat = 480  # Standard MIDI ticks per quarter note
    quarter_note_ticks = ticks_per_beat
    
    # Define representative multipliers for each duration category (indices 0-8)
    # These values can be viewed as approximate fractions or multiples of a quarter note.
    duration_multipliers = [0.125, 0.375, 0.625, 0.875, 1.25, 1.75, 3.0, 4.0, 4.0]
    
    # For each generated note, decode the duration and add MIDI messages.
    # Note: In mido, the 'time' field represents the delta time (in ticks) since the previous message.
    for pitch, dur_idx in zip(pitches, durations):
        if pitch > 127:
            pitch = 127
        # Calculate duration in ticks using the representative multiplier
        duration_ticks = int(duration_multipliers[dur_idx] * quarter_note_ticks)
        # Add a note_on message (with delta time 0)
        track.append(Message('note_on', note=pitch, velocity=64, time=0))
        # Follow with a note_off message; its delta time is the note's duration
        track.append(Message('note_off', note=pitch, velocity=64, time=duration_ticks))
    
    midi.save(output_file)

if __name__ == '__main__':
    generate()
