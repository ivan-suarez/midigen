from mido import MidiFile, MidiTrack, Message

from tensorflow.keras.models import load_model
import numpy as np

def generate():
    model = load_model('models/model_0.3.keras')

    starting_sequence = [(0.25, 0.1177083333333333, 67, 56),
    (0.007291666666666696, 0.1177083333333333, 72, 60),
    (0.007291666666666696, 0.1177083333333333, 76, 63),
    (0.007291666666666696, 0.1177083333333333, 67, 50),
    (0.007291666666666696, 0.1177083333333333, 72, 47),
    (0.007291666666666696, 0.1177083333333333, 76, 47),
    (0.2572916666666667, 0.1177083333333333, 67, 56),
    (0.007291666666666696, 0.1177083333333333, 72, 60),
    (0.007291666666666696, 0.1177083333333333, 76, 62),
    (0.007291666666666696, 0.1177083333333333, 67, 50)]

    starting_sequence_one = starting_sequence[:10]

    seed_index = 0
    seed_sequence = starting_sequence_one # Starting with an existing sequence as seed
    num_notes_to_generate = 100  # Number of notes you want to generate
    generated_notes = generate_notes(model, seed_sequence, num_notes_to_generate)
    print(generated_notes)
    create_midi_from_notes(generated_notes)
    return generated_notes

def generate_notes(model, seed_sequence, num_notes_to_generate):
    generated_sequence = seed_sequence.copy()  # Copy the seed sequence
    for _ in range(num_notes_to_generate):
        # Reshape the sequence to match the model's input shape: (1, 50, 4)
        sequence_length = 10
        input_sequence = np.array(generated_sequence[-sequence_length :]).reshape(1, sequence_length , 4)
        
        # Predict the next note
        predicted_note = model.predict(input_sequence)[0]  # model.predict returns a batch, so get the first
        
        # Append the predicted note to the sequence
        generated_sequence.append(predicted_note.tolist())
        
    return generated_sequence

# Assuming `x_train` is your training data and `seed_index` is some index of your choice3
#seed_index = 0
#seed_sequence = starting_sequence # Starting with an existing sequence as seed
#num_notes_to_generate = 100  # Number of notes you want to generate

# Generate notes
#generated_notes = generate_notes(model, seed_sequence, num_notes_to_generate)

def create_midi_from_notes(notes, output_file='generated_sequence.mid'):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Assuming the first event happens at time 0
    last_event_time = 0
    
    for note in notes:
        print(note)
        start_time, duration, pitch, velocity = map(float, note)
        print(duration)
        duration = abs(duration*10)
        end_time = start_time +  duration
        time_step = 0.25
        start_time = round(start_time / time_step) * time_step
        end_time = round(end_time / time_step) * time_step
        # Ensure non-negative delta times
        #delta_time_on = max(0, start_time - last_event_time)
        #delta_time_off = max(0, end_time - start_time)
        # Assuming a standard conversion, adjust based on your actual data and tempo
        milliseconds_per_tick = 5 / midi.ticks_per_beat  # For 120 BPM and 480 TPB
        delta_time_on = int(max(0, start_time - last_event_time)/milliseconds_per_tick)
        delta_time_off = int(max(0, end_time - start_time)/milliseconds_per_tick)

        
        # Update last event time for the next iteration
        # For note_on, this is the start time of the current note
        last_event_time = start_time
        print(start_time)
        print(duration)
        print(end_time)
        # Add the note_on message
        track.append(Message('note_on', note=int(pitch), velocity=int(velocity), time=delta_time_on))
        # For note_off, update the last event time to be the end time of the current note
        last_event_time = end_time
        
        # Add the note_off message with time since the note_on
        track.append(Message('note_off', note=int(pitch), velocity=0, time=delta_time_off))
    
    midi.save(output_file)
  #  return output_file

generate()




