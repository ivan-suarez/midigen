from mido import MidiFile, MidiTrack, Message
from tensorflow.keras.models import load_model
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
from keras.layers import Attention, LayerNormalization  # Import the custom layers we used

def generate():
    # Load the multi-output model saved from training
    # Register custom objects to ensure proper loading
    custom_objects = {
        'Attention': Attention,
        'LayerNormalization': LayerNormalization
    }
    
    model = load_model('models/LSTM_multi_output.keras', 
                      compile=False, 
                      safe_mode=False,
                      custom_objects=custom_objects)
    
    # Recompile the model
    model.compile(optimizer='adam', 
                 loss={'pitch_output': 'categorical_crossentropy',
                       'duration_output': 'categorical_crossentropy'},
                 metrics=['accuracy'])
    
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
    
    # Define a mapping from duration categories to actual durations (in quarter notes)
    # This should match the bins used in quantize_duration in main.py
    duration_mapping = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]  # 32nd, 16th, 8th, dotted 8th, quarter, etc.
    
    # Track the generated durations for debugging
    generated_duration_categories = []
    
    for _ in range(num_notes_to_generate):
        # Prepare the input: take the last 'sequence_length' notes and reshape for prediction
        input_seq = np.array(generated_sequence[-sequence_length:]).reshape(1, sequence_length, -1)
        # The model outputs two predictions: one for pitch and one for duration
        pitch_pred, duration_pred = model.predict(input_seq)
        
        # Apply different temperature scaling for pitch and duration
        # Higher temperature for pitch (more variety)
        pitch_temperature = 1.2
        # Lower temperature for duration (more structure)
        duration_temperature = 0.8
        
        # Apply temperature to pitch prediction
        pitch_pred = np.log(pitch_pred + 1e-10) / pitch_temperature
        pitch_pred = np.exp(pitch_pred)
        pitch_pred = pitch_pred / np.sum(pitch_pred)
        
        # Apply temperature to duration prediction
        duration_pred = np.log(duration_pred + 1e-10) / duration_temperature
        duration_pred = np.exp(duration_pred)
        duration_pred = duration_pred / np.sum(duration_pred)
        
        # Sample from the distributions instead of taking argmax
        pred_pitch_index = np.random.choice(range(num_pitches), p=pitch_pred[0])
        
        # For duration, we'll use a more controlled approach to ensure variety
        # Boost probabilities of longer durations slightly
        boosted_duration_pred = duration_pred[0].copy()
        for i in range(len(boosted_duration_pred)):
            if i > 1:  # Boost probabilities for durations longer than 32nd notes
                boosted_duration_pred[i] *= (1.0 + (i * 0.1))  # Progressive boost
        
        # Renormalize
        boosted_duration_pred = boosted_duration_pred / np.sum(boosted_duration_pred)
        
        # Sample duration with boosted probabilities
        pred_duration_index = np.random.choice(range(num_duration_classes), p=boosted_duration_pred)
        
        # Track the generated duration category
        generated_duration_categories.append(pred_duration_index)
        
        # Convert the predicted indices back to one-hot vectors
        pitch_vec = to_categorical(pred_pitch_index, num_classes=num_pitches)
        duration_vec = to_categorical(pred_duration_index, num_classes=num_duration_classes)
        # Concatenate to form the full feature vector for the predicted note
        new_note = np.concatenate((pitch_vec, duration_vec))
        generated_sequence.append(new_note)
    
    # Print distribution of generated duration categories for debugging
    duration_counts = np.bincount(generated_duration_categories, minlength=num_duration_classes)
    print("Duration category distribution:")
    for i, count in enumerate(duration_counts):
        if i < len(duration_mapping):
            print(f"Category {i} ({duration_mapping[i]} quarter notes): {count} notes")
        else:
            print(f"Category {i}: {count} notes")
    
    # Return only the generated notes (exclude the initial seed)
    return generated_sequence[sequence_length:]

def create_midi_from_notes(pitches, durations, output_file='generated_sequence9.mid'):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Define a mapping from duration categories to actual durations (in quarter notes)
    # This should match the bins used in quantize_duration in main.py
    duration_mapping = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]  # 32nd, 16th, 8th, dotted 8th, quarter, etc.
    
    # Set tempo (120 BPM)
    tempo = 120
    quarter_note_ticks = 480  # Standard MIDI ticks per quarter note
    
    # Assuming the first event happens at time 0
    current_time = 0
    
    # Create MIDI messages with proper timing
    for pitch, duration_category in zip(pitches, durations):
        if pitch > 127:
            pitch = 127
        
        # Map duration category to actual duration in quarter notes
        if duration_category < len(duration_mapping):
            duration_in_quarters = duration_mapping[duration_category]
        else:
            # Default to quarter note if category is out of range
            duration_in_quarters = 1.0
            
        # Convert duration to ticks
        duration_ticks = int(duration_in_quarters * quarter_note_ticks)
        
        # Add note_on event
        track.append(Message('note_on', note=pitch, velocity=64, time=0))
        # Add note_off event with appropriate duration
        track.append(Message('note_off', note=pitch, velocity=0, time=duration_ticks))
        
        current_time += duration_ticks
    
    midi.save(output_file)
    print(f"MIDI file saved as {output_file}")

if __name__ == '__main__':
    generate()
