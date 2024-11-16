import os
import json
from Bio.PDB import PDBParser
# Path settings
txt_directory = r'F:\Dropbox\新建文件夹\Dropbox\2024\Cheng lab\PDB\propedia v2.3\txt files'
json_directory = r'F:\Dropbox\新建文件夹\Dropbox\2024\Cheng lab\PDB\propedia v2.3\json_esm3'

interface_directory_path = r'F:\Dropbox\新建文件夹\Dropbox\2024\Cheng lab\PDB\propedia v2.3\interface\interface'
complex_directory_path = r'F:\Dropbox\新建文件夹\Dropbox\2024\Cheng lab\PDB\propedia v2.3\whole complex\complex\complex'
pad_length=[50, 500]

# Function to convert three letter residue name to one letter
def three_to_one_letter(three_letter_residue):
    # Dictionary mapping three-letter codes to one-letter codes
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "SEC": "U", "PYL": "O"
    }
    
    one_letter_residue = three_to_one.get(three_letter_residue, '<unk>')
    return one_letter_residue
# Function to extract sequence from a PDB file
def extract_sequence_individualChains(pdb_path, chains):
    sequences = {}
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure_id', pdb_path)

        # Get all chain IDs in the model
        model_chains = {chain.id for model in structure for chain in model}

        # Check if Individual_chains chains are in model chains
        chain_exist = [chain in model_chains for chain in chains]
        chains = [chain.swapcase() if not exist else chain for chain, exist in zip(chains, chain_exist)]
        for idx, chain_id in enumerate(chains):
            for chain in structure[0]:
                if chain.id==chain_id:
                    chain_id = chain_id.swapcase() if not chain_exist[idx] else chain_id
                    if chain_id not in sequences:
                        sequences[chain_id] = []
                    for residue in chain:
                        residue_name = three_to_one_letter(residue.resname)
                        sequences[chain_id].append(residue_name)
        #Return the sequences as a dictionary of joined strings
        return {chain: ' '.join(sequences[chain]) for chain in sequences if sequences[chain]}
    except FileNotFoundError:
        print(f"File not found: {pdb_path}")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

def extract_sequence_posIdx(pdb_path, chains):
    sequences = {}
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure_id', pdb_path)

        # Get all chain IDs in the model
        model_chains = {chain.id for model in structure for chain in model}

        # Check if Individual_chains chains are in model chains
        chain_exist = [chain in model_chains for chain in chains]
        chains = [chain.swapcase() if not exist else chain for chain, exist in zip(chains, chain_exist)]
        for idx, chain_idIndividual_chains in enumerate(chains):
                for chain in structure[0]:
                    if chain.id==chain_idIndividual_chains:
                        chain_id = chain_idIndividual_chains.swapcase() if not chain_exist[idx] else chain_idIndividual_chains
                        if chain_id not in sequences:
                            sequences[chain_id] = []
                        for residue in chain:
                            #residue_name = three_to_one_letter(residue.resname)
                            residue_index = str(residue.id[1])
                            sequences[chain_id].append(residue_index)
        # Return the sequences as a dictionary of joined strings
        return {chain: ' '.join(sequences[chain]) for chain in sequences if sequences[chain]}
    except FileNotFoundError:
        print(f"File not found: {pdb_path}")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
    
def extract_sequence_combinedChains(pdb_path, chains, pad_length):
    lenCheck_flag = True # expect lenth less than pad_length
    sequences = {}
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure_id', pdb_path)

        # Get all chain IDs in the model
        model_chains = {chain.id for model in structure for chain in model}

        # Check if Individual_chains chains are in model chains
        chain_exist = [chain in model_chains for chain in chains]
        chains = [chain.swapcase() if not exist else chain for chain, exist in zip(chains, chain_exist)]
        for idx, chain_idIndividual_chains in enumerate(chains):
                for chain in structure[0]:
                    if chain.id==chain_idIndividual_chains:
                        chain_id = chain_idIndividual_chains.swapcase() if not chain_exist[idx] else chain_idIndividual_chains
                        if chain_id not in sequences:
                            sequences[chain_id] = []
                        for residue in chain:
                            residue_name = three_to_one_letter(residue.resname)
                            sequences[chain_id].append(residue_name)
        #pad chains
        for (chain_id, chain_value), length in zip(sequences.items(), pad_length):
            part_list = chain_value
            if len(part_list)>length:
                lenCheck_flag = False
                return {}, lenCheck_flag
            while len(part_list)<length:
                part_list.append('<pad>')
            chain_value=part_list
            sequences[chain_id]=chain_value
        # Return the sequences as a dictionary of joined strings
        return {chain: ''.join(sequences[chain]) for chain in sequences if sequences[chain]}, lenCheck_flag
    except FileNotFoundError:
        print(f"File not found: {pdb_path}")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
    
def generate_binary_string(pdb_id, str_wholeComplex, str_interface, pad_length):
    binary_string = []
    str_wholeComplex_parts = [list(map(int, part.split())) for part in str_wholeComplex.split('|')]
    str_interface_parts = [list(map(int, part.split())) for part in str_interface.split('|')]

    for i1_part, i2_part in zip(str_wholeComplex_parts, str_interface_parts):
        binary_part = ['1' if pos in i2_part else '0' for pos in i1_part]
        binary_string.append(' '.join(binary_part))
    for idx, (part, length) in enumerate(zip(binary_string, pad_length)):
        part_list = part.split()
        num_padding = length - len(part_list)
        if num_padding < 0:
            print("Part length specified is less than the number of elements in the part.")
        padded_part = part_list + ['2'] * num_padding
        binary_string[idx] = ' '.join(padded_part)
    return ' 3 '.join(binary_string)

def count_lines_in_file(file_path):
    line_count = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line_count += 1
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    return line_count

numLenProblem = 0 #count number of cases that chain length less than pad length requirement
for filename in os.listdir(txt_directory):
    if filename.endswith('.txt'):
        txt_filePath = os.path.join(txt_directory, filename)
        json_filePath = os.path.join(json_directory, filename.replace('.txt', '.json'))
        pdb_dict = {}

    # Read the text file and update dictionary structure
    with open(txt_filePath, 'r') as file:
        line_count = count_lines_in_file(txt_filePath)
        sss=1

        for line in file:
            filename = line.strip().replace('.pdb', '')
            pdb_id, chains = filename.split('_')[0], filename.split('_')[1:]
            chains_tuple = tuple(chains)  # Create a tuple for consistent ordering

            # Ensure the PDB ID is in the dictionary and properly initialized
            if pdb_id not in pdb_dict:
                pdb_dict[pdb_id] = {'individual_chains': {chain: ' ' for chain in chains_tuple}, 'combined_chains':{},'posIdx_wholeComplex':{}, 'posIdx_interface': {}, 'posIdx_binary':{}}

            # Form Individual Chains Section
            chains_to_extract = [chain for chain in chains_tuple if not pdb_dict[pdb_id]['individual_chains'].get(chain)]
            complex_file_path = os.path.join(complex_directory_path, filename + '.pdb')
            
            # Extract sequences only for chains that have not been processed yet
            # if chains_to_extract:
            #     chain_sequences = extract_sequence_individualChains(complex_file_path, chains_to_extract)
            #     #chain_sequences = extract_sequence_individualChains(complex_file_path, chains_to_extract)
            #     for chain in chains_to_extract:
            #         pdb_dict[pdb_id]['individual_chains'][chain] = chain_sequences.get(chain, '')
            # Extract interaction Chains

            # Form PosIdx Section
            key = '_'.join(chains)

            if key not in pdb_dict[pdb_id]['combined_chains']:
                pdb_file_path = os.path.join(complex_directory_path, filename + '.pdb')
                sequence, lenCheck_flag = extract_sequence_combinedChains(pdb_file_path, chains_tuple, pad_length)
                if lenCheck_flag == False:
                    numLenProblem +=1
                    continue # pdb without meeting length requirement don't have any contents
                pdb_dict[pdb_id]['combined_chains'][key] = f"{'|'.join(sequence[chain] for chain in chains_tuple if chain in sequence)}"
            
            if key not in pdb_dict[pdb_id]['posIdx_interface']:
                pdb_file_path = os.path.join(interface_directory_path, filename + '.pdb')
                sequence = extract_sequence_posIdx(pdb_file_path, chains_tuple)
                pdb_dict[pdb_id]['posIdx_interface'][key] = ' | '.join(sequence[chain] for chain in chains_tuple if chain in sequence)

            if key not in pdb_dict[pdb_id]['posIdx_wholeComplex']:
                pdb_file_path = os.path.join(complex_directory_path, filename + '.pdb')
                sequence = extract_sequence_posIdx(pdb_file_path, chains_tuple)
                pdb_dict[pdb_id]['posIdx_wholeComplex'][key] = ' | '.join(sequence[chain] for chain in chains_tuple if chain in sequence)

            if key not in pdb_dict[pdb_id]['posIdx_binary']:
                str_wholeComplex=pdb_dict[pdb_id]['posIdx_wholeComplex'][key]
                str_interface=pdb_dict[pdb_id]['posIdx_interface'][key]
                str_binary = generate_binary_string(pdb_id, str_wholeComplex, str_interface, pad_length)
                pdb_dict[pdb_id]['posIdx_binary'][key]=str_binary

            print(f"Complete {sss} out of {line_count}\n")
            sss=sss+1

    for pdb_id in list(pdb_dict.keys()):
        if 'individual_chains' in pdb_dict[pdb_id]:
            del pdb_dict[pdb_id]['individual_chains']
        if 'posIdx_wholeComplex' in pdb_dict[pdb_id]:
            del pdb_dict[pdb_id]['posIdx_wholeComplex']
        if 'posIdx_interface' in pdb_dict[pdb_id]:
            del pdb_dict[pdb_id]['posIdx_interface']
        if pdb_dict[pdb_id]['combined_chains'] == {}:
            del pdb_dict[pdb_id]

    # Write updated dictionary to a JSON file
    with open(json_filePath, 'w') as json_file:
        json.dump(pdb_dict, json_file, indent=4)
print(numLenProblem)