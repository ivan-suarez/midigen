from mido import MidiFile, MidiTrack, Message

from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from keras.utils import to_categorical

def generate():
    model = load_model('models/LSTM_0.6.0.keras')

    num_pitches = 128  # MIDI pitches range from 0 to 127
    seed_pitches = [56, 60, 63, 50, 47, 47, 56, 60, 62, 50]  # Example seed sequence pitches

    # One-hot encode the seed sequence
    seed_sequence = to_categorical(seed_pitches, num_classes=num_pitches)



    num_notes_to_generate = 100  # Number of notes you want to generate
    generated_notes = generate_notes(model, seed_sequence, num_notes_to_generate)
   # print(generated_notes)

    # Convert from one-hot encoded notes back to pitches
    generated_pitches = [np.argmax(note) for note in generated_notes]

    # Here you would convert these pitches back into a MIDI file
    create_midi_from_notes(generated_pitches)

    return generated_pitches

def generate_notes(model, seed_sequence, num_notes_to_generate):
    generated_sequence = seed_sequence.tolist()  # Copy the seed sequence
    for _ in range(num_notes_to_generate):
        # Reshape the sequence to match the model's input shape: (1, 50, 4)
        input_sequence = np.array(generated_sequence[-10:]).reshape(1, 10, 128)  # Shape for LSTM (1, sequence_length, features)
        predicted_note = model.predict(input_sequence)[0]
        generated_sequence.append(predicted_note) 
        
    return generated_sequence

# Assuming `x_train` is your training data and `seed_index` is some index of your choice3
#seed_index = 0
#seed_sequence = starting_sequence # Starting with an existing sequence as seed
#num_notes_to_generate = 100  # Number of notes you want to generate

# Generate notes
#generated_notes = generate_notes(model, seed_sequence, num_notes_to_generate)

def create_midi_from_notes(pitches, output_file='generated_sequence.mid'):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Assuming the first event happens at time 0
    last_event_time = 0
    
    # Example MIDI message with fixed velocity and timing
    for pitch in pitches:
        track.append(Message('note_on', note=pitch, velocity=64, time=0))
        track.append(Message('note_off', note=pitch, velocity=0, time=480))  # Placeholder timing
    midi.save(output_file)
  #  return output_file

generate()




