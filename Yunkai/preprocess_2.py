import os
import json
import time
import sys
from Bio import PDB

def parse_pdb_file(file_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('', file_path)
    chains = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            sequence = []
            for residue in chain:
                # 这里不跳过任何残基，包括异质原子
                res_name = residue.resname
                res_id = residue.id[1]
                sequence.append(f"{res_name}:{res_id}")
            chains[chain_id] = ' '.join(sequence)
    return chains

def process_dataset(directory):
    data = {}
    start_time = time.time()
    file_list = [f for f in os.listdir(directory) if f.endswith(".pdb")]
    total_files = len(file_list)
    
    for i, file_name in enumerate(file_list):
        file_path = os.path.join(directory, file_name)
        parts = file_name.split('_')
        pdb_id = parts[0]
        peptide_chain_id = parts[1]
        protein_chain_id = parts[2].split('.')[0]
        
        chains = parse_pdb_file(file_path)
        if peptide_chain_id in chains and protein_chain_id in chains:
            peptide_sequence = chains[peptide_chain_id]
            protein_sequence = chains[protein_chain_id]
            
            peptide_res_ids = [res.split(':')[1] for res in peptide_sequence.split()]
            protein_res_ids = [res.split(':')[1] for res in protein_sequence.split()]
            
            data[pdb_id] = {
                "Input": {
                    peptide_chain_id: peptide_sequence,
                    protein_chain_id: protein_sequence
                },
                "Output": {
                    f"{peptide_chain_id}_{protein_chain_id}": f"{' '.join(peptide_res_ids)}:{' '.join(protein_res_ids)}"
                }
            }

        # 计算并输出进度
        progress = (i + 1) / total_files * 100
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / (i + 1) * total_files
        estimated_remaining_time = estimated_total_time - elapsed_time
        sys.stdout.write(f"\rProcessing: {progress:.2f}% complete. Estimated remaining time: {estimated_remaining_time:.2f} seconds")
        sys.stdout.flush()
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_file = total_time / total_files if total_files > 0 else 0
    
    return data, total_time, avg_time_per_file

def main():
    directory = "./interface"
    output_file = "./output.json"
    
    data, total_time, avg_time_per_file = process_dataset(directory)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"\nTotal time taken: {total_time:.2f} seconds")
    print(f"Average time per file: {avg_time_per_file:.2f} seconds")

if __name__ == "__main__":
    main()
