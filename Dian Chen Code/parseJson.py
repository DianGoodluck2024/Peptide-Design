import os
import json
from Bio.PDB import PDBParser
# Path settings
Individual_chains_file_path = r'A:\Research\Cheng lab\PDB\propedia v2.3\interfaceNames_test.txt'
output_json_path = r'A:\Research\Cheng lab\PDB\propedia v2.3\json\pdb_ids.json'
interface_directory_path = r'A:\Research\Cheng lab\PDB\propedia v2.3\interface'
complex_directory_path = r'A:\Research\Cheng lab\PDB\propedia v2.3\whole complex\complex'

# Function to convert three letter residue name to one letter
def three_to_one_letter(three_letter_residue):
    # Dictionary mapping three-letter codes to one-letter codes
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
    }
    
    one_letter_residue = three_to_one.get(three_letter_residue, 'X')
    return one_letter_residue
# Function to extract sequence from a PDB file
def extract_sequence_individualChains(pdb_path, chains, firstChain_maxLength, secondChain_maxLength):
# def extract_sequence_individualChains(pdb_path, chains):
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
            chain_Length=0
            for chain in structure[0]:
                if chain.id==chain_id:
                    chain_id = chain_id.swapcase() if not chain_exist[idx] else chain_id
                    if chain_id not in sequences:
                        sequences[chain_id] = []
                    for residue in chain:
                        residue_name = three_to_one_letter(residue.resname)
                        sequences[chain_id].append(residue_name)
                        chain_Length=chain_Length+1
            if idx == 0:
               firstChain_maxLength=max(firstChain_maxLength,chain_Length)
            elif idx==1:
                secondChain_maxLength=max(secondChain_maxLength,chain_Length)
        #Return the sequences as a dictionary of joined strings
        return firstChain_maxLength, secondChain_maxLength, {chain: ' '.join(sequences[chain]) for chain in sequences if sequences[chain]}
        # return {chain: ' '.join(sequences[chain]) for chain in sequences if sequences[chain]}
    except FileNotFoundError:
        print(f"File not found: {pdb_path}")
        return firstChain_maxLength, secondChain_maxLength, {}
        # return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return firstChain_maxLength, secondChain_maxLength, {}
        # return {}
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
def extract_sequence_combinedChains(pdb_path, chains):
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
                            #residue_index = str(residue.id[1])
                            sequences[chain_id].append(residue_name)
        # Return the sequences as a dictionary of joined strings
        return {chain: ' '.join(sequences[chain]) for chain in sequences if sequences[chain]}
    except FileNotFoundError:
        print(f"File not found: {pdb_path}")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
def generate_binary_string(str_wholeComplex, str_interface):
    binary_string = []
    str_wholeComplex_parts = [list(map(int, part.split())) for part in str_wholeComplex.split('|')]
    str_interface_parts = [list(map(int, part.split())) for part in str_interface.split('|')]

    for i1_part, i2_part in zip(str_wholeComplex_parts, str_interface_parts):
        binary_part = ['1' if pos in i2_part else '0' for pos in i1_part]
        binary_string.append(' '.join(binary_part))

    return ' | '.join(binary_string)
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


# Initialize or load existing dictionary
if os.path.exists(output_json_path):
    with open(output_json_path, 'r') as file:
        pdb_dict = json.load(file)
else:
    pdb_dict = {}

# Read the text file and update dictionary structure
with open(Individual_chains_file_path, 'r') as file:
    line_count = count_lines_in_file(Individual_chains_file_path)
    sss=1
    firstChain_maxLength=0
    secondChain_maxLength=0
    for line in file:
        filename = line.strip().replace('.pdb', '')
        pdb_id, chains = filename.split('_')[0], filename.split('_')[1:]
        chains_tuple = tuple(chains)  # Create a tuple for consistent ordering

        # Ensure the PDB ID is in the dictionary and properly initialized
        if pdb_id not in pdb_dict:
            pdb_dict[pdb_id] = {'individual_chains': {chain: '' for chain in chains_tuple}, 'combined_chains':{},'posIdx_wholeComplex':{}, 'posIdx_interface': {}, 'posIdx_binary':{}}

        # Form Individual Chains Section
        chains_to_extract = [chain for chain in chains_tuple if not pdb_dict[pdb_id]['individual_chains'].get(chain)]
        complex_file_path = os.path.join(complex_directory_path, filename + '.pdb')
        
        # Extract sequences only for chains that have not been processed yet
        if chains_to_extract:
            firstChain_maxLength, secondChain_maxLength, chain_sequences = extract_sequence_individualChains(complex_file_path, chains_to_extract, firstChain_maxLength, secondChain_maxLength)
            #chain_sequences = extract_sequence_individualChains(complex_file_path, chains_to_extract)
            for chain in chains_to_extract:
                pdb_dict[pdb_id]['individual_chains'][chain] = chain_sequences.get(chain, '')
        # Extract interaction Chains

        # Form PosIdx Section
        key = '_'.join(chains)

        if key not in pdb_dict[pdb_id]['combined_chains']:
            pdb_file_path = os.path.join(complex_directory_path, filename + '.pdb')
            sequence = extract_sequence_combinedChains(pdb_file_path, chains_tuple)
            pdb_dict[pdb_id]['combined_chains'][key] = ' [SEP] '.join(sequence[chain] for chain in chains_tuple if chain in sequence)
        
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
            str_binary = generate_binary_string(str_wholeComplex, str_interface)
            pdb_dict[pdb_id]['posIdx_binary'][key]=str_binary

        print(f"Complete {sss} out of {line_count}\n")
        sss=sss+1

print(f"firstChain_maxLength = {firstChain_maxLength}\n")
print(f"secondChain_maxLength = {secondChain_maxLength}\n")
#pdb_dict['max_chainLength'] = ' | '.join([str(firstChain_maxLength), str(secondChain_maxLength)])
for pdb_id in pdb_dict:
    if 'individual_chains' in pdb_dict[pdb_id]:
        del pdb_dict[pdb_id]['individual_chains']
    if 'posIdx_wholeComplex' in pdb_dict[pdb_id]:
        del pdb_dict[pdb_id]['posIdx_wholeComplex']
    if 'posIdx_interface' in pdb_dict[pdb_id]:
        del pdb_dict[pdb_id]['posIdx_interface']

# Write updated dictionary to a JSON file
with open(output_json_path, 'w') as json_file:
    json.dump(pdb_dict, json_file, indent=4)