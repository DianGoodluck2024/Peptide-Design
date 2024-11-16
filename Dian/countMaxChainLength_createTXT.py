import json
import matplotlib.pyplot as plt
import numpy as np

# Function to read JSON file
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Function to count the maximum number of residues on the first and second chains
def count_max_residues(data, len_threshold, first_chain_min_length):
    first_chain_len = []
    second_chain_len = []
    max_first_chain = 0
    max_second_chain = 0
    pdb_ids_thresholded = []
    total_pdb = 0

    sss=0
    kkk=0
    for pdb_id, pdb_data in data.items():
        output = pdb_data.get("Output", {})
        input_data = pdb_data.get("Input", {})

        for chain_key, residues in output.items():
            sss +=1
            if '|' in residues:
                first_chain_key, second_chain_key = chain_key.split('_')
                first_chain_residues = input_data.get(first_chain_key, "")
                second_chain_residues = input_data.get(second_chain_key, "")
                
                # Count residues by splitting the string
                first_chain_count = len(first_chain_residues.strip().split())
                second_chain_count = len(second_chain_residues.strip().split())

                # Append lengths for possible plotting later
                first_chain_len.append(first_chain_count)
                second_chain_len.append(second_chain_count)

                # Update the maximum residues found
                if first_chain_count > max_first_chain:
                    max_first_chain = first_chain_count
                if second_chain_count > max_second_chain:
                    max_second_chain = second_chain_count

                # Only add PDB if the first chain has at least `first_chain_min_length` residues
                # and the second chain is <= len_threshold
                if first_chain_count >= first_chain_min_length and second_chain_count < len_threshold:
                    pdb_ids_thresholded.append(pdb_id + '_' + chain_key)
                    kkk+=1

    return max_first_chain, max_second_chain, first_chain_len, second_chain_len, pdb_ids_thresholded

# Initialization
input_file_path = r'F:\Dropbox\新建文件夹\Dropbox\2024\Cheng lab\PDB\propedia v2.3\pdb_ids_original.json'
output_file_path = r'F:\Dropbox\新建文件夹\Dropbox\2024\Cheng lab\PDB\propedia v2.3\pdb_ids_afterDeletion.txt'
len_threshold = 500

# Set the minimum length for the first chain as a variable
first_chain_min_length = 9

# Read the JSON file
data = read_json(input_file_path)

# Count the maximum number of residues on the first and second chains
max_first_chain, max_second_chain, first_chain_len, second_chain_len, pdb_ids_thresholded = count_max_residues(
    data, len_threshold, first_chain_min_length
)

# Output results
print(f'Maximum number of residues on the first chain: {max_first_chain}')
print(f'Maximum number of residues on the second chain: {max_second_chain}')
print(len(pdb_ids_thresholded))

# Write filtered PDB IDs to a text file
with open(output_file_path, "w") as file:
    for item in pdb_ids_thresholded:
        file.write(item + "\n")

# Output message to indicate the file has been written
print("Data has been written to pdb_ids_lessThan500.txt")
