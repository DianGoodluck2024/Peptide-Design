from Bio import PDB
import numpy as np
import os
import json

# Atomic masses for common elements (in atomic mass units, amu)
atomic_masses = {'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'K': 39.098, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723, 'Ge': 72.63, 'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.95, 'Tc': 98, 'Ru': 101.07, 'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41, 'In': 114.82, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.60, 'I': 126.90, 'Xe': 131.29, 'Cs': 132.91, 'Ba': 137.33, 'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24, 'Pm': 145, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.50, 'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97, 'Hf': 178.49, 'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59, 'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.98, 'Th': 232.04, 'Pa': 231.04, 'U': 238.03}

def get_atomic_mass(atom):
    """
    Get the atomic mass of an atom using its element.
    """
    element = atom.element.capitalize()  # Get the element symbol (e.g., 'C', 'O')
    return atomic_masses.get(element, 12.011)  # Default to carbon if element not found

def calculate_mass_center(residue):
    """
    Calculate the mass-weighted center of mass (COM) for a residue.
    """
    total_mass = 0.0
    weighted_coords = np.zeros(3)

    # Sum the weighted coordinates by atom mass
    for atom in residue.get_atoms():
        mass = get_atomic_mass(atom)
        coord = atom.get_coord()
        weighted_coords += mass * coord
        total_mass += mass

    # Return the mass-weighted center of mass
    if total_mass == 0:
        return None  # If no atoms, return None
    com = weighted_coords / total_mass
    return com

def calculate_distances_com(chain, interface_binary, error_log, pdb_info):
    """
    Calculate distances based on the center of mass (COM) method between
    interface and non-interface residues.
    """
    distances = []
    residues = list(chain.get_residues())

    # Calculate mass-weighted COM for all residues
    residues_com = [calculate_mass_center(residue) for residue in residues]
    
    # Filter out None values (residues that might not have atoms)
    residues_com = [com for com in residues_com if com is not None]

    # Get COMs for interface residues
    interface_coms = [residues_com[i] for i, is_interface in enumerate(interface_binary) if is_interface == 1]

    if len(interface_coms) == 0:
        error_log.write(f"No interface residues found for {pdb_info}. Skipping distance calculation.\n")
        return [-1] * len(residues)  # Return a placeholder if no interface residues

    for i, is_interface in enumerate(interface_binary):
        if is_interface == 1:
            # Distance is 0 for interface residues
            distances.append(0)
        else:
            non_interface_com = residues_com[i]
            if non_interface_com is not None:
                # Calculate distances from the current non-interface residue's COM to all interface residues' COMs
                com_distances = np.linalg.norm(interface_coms - non_interface_com, axis=1)
                min_distance = np.min(com_distances)
                distances.append(min_distance)
            else:
                distances.append(-1)  # If the residue has no COM (no atoms), append -1

    return distances

def pad_list(input_list, target_length, padding_value):
    # Check if the list is longer than the target length and raise an error if it is
    if len(input_list) > target_length:
        raise ValueError(f"Input list is longer than the target length of {target_length}. Length of input list: {len(input_list)}")
    
    # Pad the list with the padding_value until the target length is reached
    return input_list + [padding_value] * (target_length - len(input_list))

# Paths for JSON folder and PDB folder
json_folder_path = r"F:\Dropbox\新建文件夹\Dropbox\2024\Cheng lab\PDB\propedia v2.3\json_esm3 - Copy"
pdb_folder_path = r"F:\Dropbox\新建文件夹\Dropbox\2024\Cheng lab\PDB\propedia v2.3\whole complex\complex\complex"
error_log_path = r"F:\Dropbox\新建文件夹\Dropbox\2024\Cheng lab\Code\error_log.txt"
scale = 10 # weight scale
# Get the list of JSON files
json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
total_files = len(json_files)

# Initialize the Biopython PDB parser and structure builder
pdb_parser = PDB.PDBParser(QUIET=True)

sss = 1
with open(error_log_path, 'a') as error_log:
    # Iterate over all JSON files in the directory
    for index, file_name in enumerate(json_files):
        file_path = os.path.join(json_folder_path, file_name)

        # Open and read the JSON file
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

            for pdb_id, content in data.items():
                for combined_chains in content['combined_chains'].keys():
                    posIdx_binary_str = content['posIdx_binary'][combined_chains]
                    posIdx_binary = list(map(int, posIdx_binary_str.split()))
                    split_index = posIdx_binary.index(3)

                    peptide_interface = posIdx_binary[:split_index]
                    protein_interface = posIdx_binary[split_index + 1:]

                    # Remove '2's as they don't indicate relevant residues
                    peptide_interface = [x for x in peptide_interface if x != 2]
                    protein_interface = [x for x in protein_interface if x != 2]

                    chain_ids = combined_chains.split('_')

                    if len(chain_ids) == 2:
                        chain1, chain2 = chain_ids

                        pdb_file_name = f"{pdb_id}_{combined_chains}.pdb"
                        pdb_file_path = os.path.join(pdb_folder_path, pdb_file_name)

                        if os.path.exists(pdb_file_path):
                            try:
                                # Load structure from PDB file
                                structure = pdb_parser.get_structure(pdb_id, pdb_file_path)

                                # Select peptide and protein chains
                                peptide_chain = structure[0][chain1]
                                protein_chain = structure[0][chain2]

                                # Validate the length of residues
                                assert len(list(peptide_chain.get_residues())) == len(peptide_interface), f"Mismatch in {file_name}_{pdb_id}_{combined_chains}"
                                assert len(list(protein_chain.get_residues())) == len(protein_interface), f"Mismatch in {file_name}_{pdb_id}_{combined_chains}"

                                pdb_info = f"{file_name}_{pdb_id}_{combined_chains}"

                                # Calculate distances using mass-weighted COM method
                                peptide_distances = calculate_distances_com(peptide_chain, peptide_interface, error_log, pdb_info)
                                protein_distances = calculate_distances_com(protein_chain, protein_interface, error_log, pdb_info)

                                # Normalize distances
                                min_distance = 0
                                peptide_max_distance = max(peptide_distances)
                                protein_max_distance = max(protein_distances)
                                if peptide_max_distance != 0:
                                    peptide_distances = [scale - scale * (d - min_distance) / (peptide_max_distance - min_distance) for d in peptide_distances]
                                if protein_max_distance != 0:
                                    protein_distances = [scale - scale * (d - min_distance) / (protein_max_distance - min_distance) for d in protein_distances]
                                # Combine the distances with a separator (-1)
                                peptide_distances = pad_list(peptide_distances, target_length = 50, padding_value = -2)
                                protein_distances = pad_list(protein_distances, target_length = 500, padding_value = -2)
                                combined_distances = peptide_distances + [-1] + protein_distances

                                if "distance" not in data[pdb_id]:
                                    data[pdb_id]["distance"] = {}

                                # Store combined distances
                                data[pdb_id]["distance"][combined_chains] = combined_distances

                            except AssertionError as e:
                                error_log.write(f"Assertion error: {e}\n")
                        else:
                            error_log.write(f"PDB file {pdb_file_name} not found in {pdb_folder_path}\n")
                    else:
                        error_log.write(f"Unexpected chain format for PDB ID {pdb_id}: {combined_chains}\n")
        # After processing, write the updated data back to the same JSON file
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)  # Write updated data back to JSON
        print(f'Complete {sss} out of {total_files}')
        sss += 1