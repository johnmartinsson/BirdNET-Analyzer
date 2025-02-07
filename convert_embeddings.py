import os
import json
import msgpack

# Define the input and output directories
input_dir = "/home/john/Gits/BirdNET-Analyzer/example/"
output_dir = "/home/john/Gits/BirdNET-Analyzer/example/"

# Define the time resolution
time_resolution = 1

# Loop over all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".birdnet.embeddings.txt"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename.replace(".txt", ".msgpack"))

        # Initialize lists to store timings and embeddings
        timings = []
        embeddings = []

        # Read the input file
        with open(input_file, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                start_time = float(parts[0])
                end_time = float(parts[1])
                embedding = list(map(float, parts[2].split(',')))

                # Adjust the timings
                adjusted_start_time = start_time + (3 - time_resolution) / 2
                adjusted_end_time = end_time - (3 - time_resolution) / 2

                # Append the adjusted timings and embeddings to the lists
                timings.append([adjusted_start_time, adjusted_end_time])
                embeddings.append(embedding)

        # Create the output dictionary
        output_data = {
            "timings": timings,
            "embeddings": embeddings
        }

        # Write the output data to the MessagePack file
        with open(output_file, 'wb') as msgpack_file:
            msgpack.pack(output_data, msgpack_file)

print("Conversion complete.")