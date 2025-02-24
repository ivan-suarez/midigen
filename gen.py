from mido import MidiFile, MidiTrack, Message

from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from keras.utils import to_categorical

def generate():
    model = load_model('models/LSTM_0.7.2.keras')

    num_pitches = 128  # MIDI pitches range from 0 to 127
    num_duration_classes = 8  # Number of duration categories
    seed_pitches = [56, 60, 63, 50, 47, 47, 56, 60, 62, 50]  # Example seed sequence pitches
    seed_durations = to_categorical([4] * 10, num_classes=num_duration_classes)  # Index 4 could correspond to quarter notes

    # One-hot encode the seed sequence
    seed_sequence = to_categorical(seed_pitches, num_classes=num_pitches)
    seed_sequence = np.concatenate((seed_sequence, seed_durations), axis=1)  # Combine pitch and duration features


    num_notes_to_generate = 100  # Number of notes you want to generate
    generated_notes = generate_notes(model, seed_sequence, num_notes_to_generate)
    
   # print(generated_notes)

    # Convert from one-hot encoded notes back to pitches
    generated_pitches = [np.argmax(note[:num_pitches]) for note in generated_notes]
    generated_durations = [np.argmax(note[num_pitches:num_pitches + num_duration_classes]) for note in generated_notes]


    # Here you would convert these pitches back into a MIDI file
    create_midi_from_notes(generated_pitches, generated_durations)

    return generated_pitches, generated_durations

def generate_notes(model, seed_sequence, num_notes_to_generate):
    generated_sequence = seed_sequence.tolist()  # Copy the seed sequence
    for _ in range(num_notes_to_generate):
        # Reshape the sequence to match the model's input shape: (1, 50, 4)
        input_sequence = np.array(generated_sequence[-10:]).reshape(1, 10, -1)  # Shape for LSTM (1, sequence_length, features)
        predicted_note = model.predict(input_sequence)[0]
        generated_sequence.append(predicted_note) 
        
    return generated_sequence[10:]

# Assuming `x_train` is your training data and `seed_index` is some index of your choice3
#seed_index = 0
#seed_sequence = starting_sequence # Starting with an existing sequence as seed
#num_notes_to_generate = 100  # Number of notes you want to generate

# Generate notes
#generated_notes = generate_notes(model, seed_sequence, num_notes_to_generate)

def create_midi_from_notes(pitches, durations, output_file='generated_sequence6.mid'):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Assuming the first event happens at time 0
    last_event_time = 0
    bins = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, np.inf]
    # Example MIDI message with fixed velocity and timing
    for pitch, duration in zip(pitches, durations):
        if(pitch>127):
            pitch=127
        tempo = 120
        quarter_note_duration = 60 / tempo 
        decoded_duration = bins[duration]* quarter_note_duration
        track.append(Message('note_on', note=pitch, velocity=64, time=last_event_time))
        track.append(Message('note_off', note=pitch, velocity=0, time=int(last_event_time+decoded_duration)))  # Placeholder timing
        last_event_time+=duration
    midi.save(output_file)
  #  return output_file

generate()




