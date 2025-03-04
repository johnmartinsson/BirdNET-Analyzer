import os
import sys
import argparse
import librosa
import soundfile as sf
import numpy as np
import subprocess
import msgpack
import matplotlib.pyplot as plt
import librosa.display
import json

import os
import librosa
import soundfile as sf

# Check librosa version
librosa_version = librosa.__version__

def get_audio_duration(audio_segment_path):
    """Get the duration of an audio segment."""
    if librosa_version >= '0.10':
        return librosa.get_duration(path=audio_segment_path)
    else:
        return librosa.get_duration(filename=audio_segment_path)


def split_audio_files(input_dir, output_audio_dir, segment_duration=30):
    """Splits WAV audio files into segments of specified duration."""
    os.makedirs(output_audio_dir, exist_ok=True)
    original_audio_files = {}  # Store original filenames and segment counts
    for filename in os.listdir(input_dir):
        if filename.endswith(".WAV") or filename.endswith(".wav"):
            print("Splitting: ", filename)
            input_audio_path = os.path.join(input_dir, filename)
            segment_filenames = []
            try:
                y, sr = librosa.load(input_audio_path, sr=None)  # Load audio with original sample rate
                segment_samples = segment_duration * sr
                num_segments = max(1, len(y) // segment_samples)  # Ensure at least one segment
                original_audio_files[filename] = {'num_segments': num_segments, 'segments_list': []}

                for i in range(num_segments):
                    start_sample = i * segment_samples
                    end_sample = (i + 1) * segment_samples if (i + 1) * segment_samples < len(y) else len(y)
                    segment = y[start_sample:end_sample]

                    # Generate segment filename
                    base_filename = os.path.splitext(filename)[0]  # remove extension
                    output_segment_filename = f"{base_filename}_segment_{i+1}.wav"
                    output_segment_path = os.path.join(output_audio_dir, output_segment_filename)
                    if not os.path.exists(output_segment_path):
                        sf.write(output_segment_path, segment, sr)
                        print(f"Saved segment {output_segment_filename}")
                    segment_filenames.append(output_segment_filename)
                original_audio_files[filename]['segments_list'] = segment_filenames

            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return original_audio_files

def generate_birdnet_embeddings(input_audio_dir, output_embedding_dir):
    """Generates BirdNET embeddings for audio files in input_audio_dir."""
    os.makedirs(output_embedding_dir, exist_ok=True)
    command = [
        "python", "-m", "birdnet_analyzer.embeddings",
        "--i", input_audio_dir,
        "--o", output_embedding_dir,
        "--threads", "8",
        "--batchsize", "1",
        "--overlap", "2.5"
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, output=process.stdout, stderr=process.stderr)
        print(f"BirdNET embeddings generated in {output_embedding_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error running BirdNET Analyzer: {e.stderr.read()}")
    except FileNotFoundError:
        print("Error: birdnet_analyzer not found. Ensure it is installed and in your PATH.")

def convert_embeddings_to_msgpack(input_embedding_dir):
    """Converts BirdNET embedding text files to msgpack format."""
    embedding_files_map = {}
    for filename in os.listdir(input_embedding_dir):
        if filename.endswith(".birdnet.embeddings.txt"):
            input_file = os.path.join(input_embedding_dir, filename)
            output_file = os.path.join(input_embedding_dir, filename.replace(".txt", ".msgpack"))
            base_audio_filename = filename.replace(".birdnet.embeddings.txt", ".wav") # Assumes embedding filename is based on audio

            if not os.path.exists(output_file):
                timings = []
                embeddings = []

                with open(input_file, 'r') as file:
                    for line in file:
                        parts = line.strip().split('\t')
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        embedding = list(map(float, parts[2].split(',')))

                        adjusted_start_time = start_time + (3 - 1) / 2 # time_resolution = 1
                        adjusted_end_time = end_time - (3 - 1) / 2

                        timings.append([adjusted_start_time, adjusted_end_time])
                        embeddings.append(embedding)

                output_data = {
                    "timings": timings,
                    "embeddings": embeddings
                }

                with open(output_file, 'wb') as msgpack_file:
                    msgpack.pack(output_data, msgpack_file)
                print(f"Converted embeddings to msgpack: {output_file}")
                os.remove(input_file) # Clean up text file
            embedding_files_map[base_audio_filename] = os.path.basename(output_file) # Store msgpack filename
    return embedding_files_map

def save_melspectrogram(audio_path, output_path, n_fft=2048, hop_length=1024, n_mels=128, figsize=(15, 3)):
    """Saves a mel spectrogram as a PNG image."""
    y, sr = librosa.load(audio_path, sr=None) # Load audio with original sample rate
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=figsize)
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='gray_r')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True) # transparent background
    plt.close()

def generate_spectrograms(input_audio_dir, output_spectrogram_dir):
    """Generates and saves spectrograms for audio files."""
    spectrogram_files_map = {}
    os.makedirs(output_spectrogram_dir, exist_ok=True)
    for filename in os.listdir(input_audio_dir):
        if filename.endswith('.wav'):
            audio_path = os.path.join(input_audio_dir, filename)
            output_path = os.path.join(output_spectrogram_dir, filename.replace('.wav', '.png'))
            if not os.path.exists(output_path):
                save_melspectrogram(audio_path, output_path)
                print(f"Saved mel spectrogram for {filename} to {output_path}")
            spectrogram_files_map[filename] = os.path.basename(output_path) # Store spectrogram filename
    return spectrogram_files_map

def create_metadata_json(output_dir, audio_dir, embedding_dir, spectrogram_dir, original_audio_info, embedding_files_map, spectrogram_files_map, sample_rate, segment_length):
    """Creates metadata.json file in the output directory."""
    metadata = {
        "audio_dir": audio_dir,
        "embedding_dir": embedding_dir,
        "spectrogram_dir": spectrogram_dir,
        "embedding_size": 1024,
        "files": {
            "audio_files": [],
            "audio_lengths": {},
            "embedding_files": [],
            "spectrogram_files": []
        },
        "embedding_model": "birdnet",
        "sample_rate": sample_rate, # Use dynamically determined sample rate
        "segment_length": segment_length # Use segment_length from processing
    }

    all_audio_files = []
    audio_lengths = {}
    all_embedding_files = []
    all_spectrogram_files = []

    for original_filename, info in original_audio_info.items():
        for segment_filename in info['segments_list']:
            audio_segment_path = os.path.join(audio_dir, segment_filename)
            segment_duration = get_audio_duration(audio_segment_path)
            all_audio_files.append(segment_filename)
            audio_lengths[segment_filename] = segment_duration
            if segment_filename in embedding_files_map:
                all_embedding_files.append(embedding_files_map[segment_filename])
            if segment_filename in spectrogram_files_map:
                all_spectrogram_files.append(spectrogram_files_map[segment_filename])

    metadata["files"]["audio_files"] = all_audio_files
    metadata["files"]["audio_lengths"] = audio_lengths
    metadata["files"]["embedding_files"] = all_embedding_files
    metadata["files"]["spectrogram_files"] = all_spectrogram_files

    metadata_filepath = os.path.join(output_dir, "metadata.json")
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_filepath}")


def main():
    parser = argparse.ArgumentParser(description="Process audio files for BirdNET analysis and spectrogram generation.")
    parser.add_argument("input_audio_dir", help="Path to the input directory containing WAV audio files.")
    parser.add_argument("output_dir", help="Path to the output directory where results will be saved.")

    args = parser.parse_args()

    output_audio_segments_dir = os.path.join(args.output_dir, "audio")
    output_embedding_dir = os.path.join(args.output_dir, "embeddings")
    output_spectrogram_dir = os.path.join(args.output_dir, "spectrograms")

    # create directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(output_audio_segments_dir, exist_ok=True)
    os.makedirs(output_embedding_dir, exist_ok=True)
    os.makedirs(output_spectrogram_dir, exist_ok=True)

    print("Splitting audio files into 30-second segments...")
    original_audio_info = split_audio_files(args.input_audio_dir, output_audio_segments_dir)

    print("Generating BirdNET embeddings...")
    if not any(fname.endswith(".msgpack") for fname in os.listdir(output_embedding_dir)):
        generate_birdnet_embeddings(output_audio_segments_dir, output_embedding_dir)

    print("Converting embeddings to msgpack format...")
    embedding_files_map = convert_embeddings_to_msgpack(output_embedding_dir)

    print("Generating spectrograms...")
    spectrogram_files_map = generate_spectrograms(output_audio_segments_dir, output_spectrogram_dir)

    print("Creating metadata file...")
    # read sample rate from a sample audio file
    print("Output audio segments dir: ", output_audio_segments_dir)
    waveform, sample_rate = librosa.load(os.path.join(output_audio_segments_dir, os.listdir(output_audio_segments_dir)[0]), sr=None)
    segment_length = len(waveform) / sample_rate

    create_metadata_json(args.output_dir, output_audio_segments_dir, output_embedding_dir, output_spectrogram_dir, original_audio_info, embedding_files_map, spectrogram_files_map, sample_rate, segment_length)

    print("Audio processing complete.")

if __name__ == "__main__":
    main()