import os

# Path to the input file
input_file = r'F:\Dropbox\新建文件夹\Dropbox\2024\Cheng lab\PDB\propedia v2.3\pdb_ids_afterDeletion.txt'
# Directory to store the output files
output_dir = r'F:\Dropbox\新建文件夹\Dropbox\2024\Cheng lab\PDB\propedia v2.3\txt files'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read the input file
with open(input_file, 'r') as file:
    lines = file.readlines()

# Determine the number of lines per output file
total_lines = len(lines)
lines_per_file = total_lines // 100
remainder = total_lines % 100

# Split and write to output files
start = 0
for i in range(100):
    end = start + lines_per_file + (1 if i < remainder else 0)
    with open(os.path.join(output_dir, f'pdb{i + 1}.txt'), 'w') as out_file:
        out_file.writelines(lines[start:end])
    start = end
