import os
import shutil
import pretty_midi
import numpy as np

def transpose_midi(input_file, output_file, semitones):
    """
    Transpose a MIDI file by the specified number of semitones.
    
    Args:
        input_file (str): Path to the input MIDI file
        output_file (str): Path to save the transposed MIDI file
        semitones (int): Number of semitones to transpose (-12 to 12)
    """
    # Load the MIDI file
    try:
        midi_data = pretty_midi.PrettyMIDI(input_file)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return False
    
    # Transpose each note in each instrument
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Transpose the pitch
            new_pitch = note.pitch + semitones
            
            # Make sure the new pitch is valid (0-127)
            if 0 <= new_pitch <= 127:
                note.pitch = new_pitch
            else:
                # If outside MIDI range, adjust octave but keep the note
                if new_pitch < 0:
                    note.pitch = (note.pitch + semitones) % 12 + (12 * (note.pitch // 12 - 1))
                else:  # new_pitch > 127
                    note.pitch = (note.pitch + semitones) % 12 + (12 * (note.pitch // 12))
    
    # Write the transposed MIDI file
    try:
        midi_data.write(output_file)
        return True
    except Exception as e:
        print(f"Error writing {output_file}: {e}")
        return False

def transpose_all_files(input_dir, output_dir):
    """
    Transpose all MIDI files in the input directory to all 12 possible keys
    and save them to the output directory.
    
    Args:
        input_dir (str): Directory containing original MIDI files
        output_dir (str): Directory to save transposed MIDI files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all MIDI files in the input directory
    midi_files = [f for f in os.listdir(input_dir) if f.endswith('.mid')]
    
    total_files = len(midi_files)
    successful_transpositions = 0
    failed_transpositions = 0
    
    print(f"Found {total_files} MIDI files in {input_dir}")
    
    # Transpose each file to all 12 possible keys
    for midi_file in midi_files:
        input_path = os.path.join(input_dir, midi_file)
        
        # First, copy the original file to the output directory
        original_output = os.path.join(output_dir, midi_file)
        try:
            shutil.copy2(input_path, original_output)
            successful_transpositions += 1
        except Exception as e:
            print(f"Error copying original file {midi_file}: {e}")
            failed_transpositions += 1
            continue
        
        # Transpose to all other keys (1 to 11 semitones)
        for semitones in range(1, 12):
            base_name, ext = os.path.splitext(midi_file)
            transposed_file = f"{base_name}_transpose_{semitones}{ext}"
            output_path = os.path.join(output_dir, transposed_file)
            
            success = transpose_midi(input_path, output_path, semitones)
            if success:
                successful_transpositions += 1
            else:
                failed_transpositions += 1
    
    print(f"Transposition complete:")
    print(f"  - Original files: {total_files}")
    print(f"  - Successful transpositions: {successful_transpositions}")
    print(f"  - Failed transpositions: {failed_transpositions}")
    print(f"  - Total files created: {successful_transpositions}")

if __name__ == "__main__":
    # Directory containing the classical MIDI files
    input_directory = "data/classical"
    
    # Directory to save the transposed files
    output_directory = "data/classical_augmented"
    
    # Transpose all files in the input directory
    transpose_all_files(input_directory, output_directory)
    
    print(f"All MIDI files have been transposed and saved to {output_directory}") 