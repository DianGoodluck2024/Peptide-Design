import os
import json
import numpy as np
import h5py
import time
import sys
from Bio import PDB
from scipy.spatial import distance

def calculate_centroid(residue):
    atoms = [atom.get_vector().get_array() for atom in residue.get_atoms()]
    centroid = np.mean(atoms, axis=0)
    return centroid

def parse_pdb_file(file_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('', file_path)
    centroids = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            centroids[chain_id] = {}
            for residue in chain:
                # 这里不跳过任何残基，包括异质原子
                res_id = residue.id[1]
                centroids[chain_id][res_id] = calculate_centroid(residue)
    return centroids

def calculate_distance_map(peptide_centroids, protein_centroids):
    peptide_ids = sorted(peptide_centroids.keys())
    protein_ids = sorted(protein_centroids.keys())
    
    distance_map = np.zeros((len(peptide_ids), len(protein_ids)))
    
    for i, pid in enumerate(peptide_ids):
        for j, prid in enumerate(protein_ids):
            distance_map[i, j] = distance.euclidean(peptide_centroids[pid], protein_centroids[prid])
    
    return distance_map

def process_and_save_to_h5(directory, json_file, h5_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    start_time = time.time()
    file_count = len(data)
    
    with h5py.File(h5_file, 'w') as h5f:
        for i, (pdb_id, pdb_data) in enumerate(data.items()):
            peptide_chain_id = list(pdb_data["Input"].keys())[0]
            protein_chain_id = list(pdb_data["Input"].keys())[1]
            pdb_file_name = f"{pdb_id}.pdb"
            pdb_file_path = os.path.join(directory, pdb_file_name)
            
            if os.path.exists(pdb_file_path):
                centroids = parse_pdb_file(pdb_file_path)
                
                peptide_centroids = centroids[peptide_chain_id]
                protein_centroids = centroids[protein_chain_id]
                
                distance_map = calculate_distance_map(peptide_centroids, protein_centroids)
                
                grp = h5f.create_group(pdb_id)
                grp.create_dataset('distance_map', data=distance_map)
            else:
                print(f"File {pdb_file_path} does not exist")
            
            # 计算并输出进度
            progress = (i + 1) / file_count * 100
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / (i + 1) * file_count
            estimated_remaining_time = estimated_total_time - elapsed_time
            sys.stdout.write(f"\rProcessing: {progress:.2f}% complete. Estimated remaining time: {estimated_remaining_time:.2f} seconds")
            sys.stdout.flush()
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_file = total_time / file_count if file_count > 0 else 0
    
    print(f"\nTotal time taken: {total_time:.2f} seconds")
    print(f"Average time per file: {avg_time_per_file:.2f} seconds")

def main():
    directory = "./complex"
    json_file = "./../interfaces2_3/output_interface.json"
    h5_file = "output.h5"
    
    process_and_save_to_h5(directory, json_file, h5_file)

if __name__ == "__main__":
    main()