import os
import json
from Bio import PDB

def extract_full_chain(pdb_file, chain_id):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('', pdb_file)
    chain = structure[0][chain_id]
    full_sequence = ' '.join([f"{residue.get_resname()}:{residue.id[1]}" for residue in chain])
    return full_sequence

def update_json_with_full_chains(json_file, pdb_folder):
    with open(json_file, 'r') as f:
        data = json.load(f)

    for pdb_id, pdb_data in data.items():
        peptide_chain_id = list(pdb_data["Input"].keys())[0]
        protein_chain_id = list(pdb_data["Input"].keys())[1]
        pdb_filename = f"{pdb_id}.pdb"
        pdb_file = os.path.join(pdb_folder, pdb_filename)

        if os.path.exists(pdb_file):
            print(f"Processing {pdb_filename}...")
            for chain_id in list(pdb_data["Input"].keys()):  # Use list() to avoid runtime dictionary size change error
                full_chain_sequence = extract_full_chain(pdb_file, chain_id)
                print(f"Original {chain_id} sequence: {pdb_data['Input'][chain_id]}")
                print(f"Updated {chain_id} sequence: {full_chain_sequence}")
                pdb_data["Input"][chain_id] = full_chain_sequence
        else:
            print(f"Warning: PDB file {pdb_file} not found!")

    updated_json_file = 'updated_' + os.path.basename(json_file)
    with open(updated_json_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Updated JSON saved as {updated_json_file}")

if __name__ == "__main__":
    json_file = 'output_interface.json'  # 替换为你的 JSON 文件名
    pdb_folder = 'complex'  # 替换为存储 PDB 文件的文件夹名
    update_json_with_full_chains(json_file, pdb_folder)
