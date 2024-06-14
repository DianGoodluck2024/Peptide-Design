import os
import csv
import json

# Function to extract PDB IDs and combined chain keys from a JSON file
def extract_pdb_ids_and_combined_chains(json_file_path):
    with open(json_file_path, 'r') as file:
        pdb_data = json.load(file)
    pdb_ids_combined_chains = []
    for pdb_id, pdb_content in pdb_data.items():
        combined_chains_keys = list(pdb_content.get('combined_chains', {}).keys())
        pdb_ids_combined_chains.append((pdb_id, combined_chains_keys))
    return pdb_ids_combined_chains

# Function to record JSON filenames, PDB IDs, and combined chain keys in a CSV file
def record_json_filenames_pdb_ids_and_combined_chains_to_csv(directory_path, csv_file_path):
    # Get list of all JSON filenames in the directory
    json_filenames = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    # Open CSV file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        #writer.writerow(['Filename', 'PDB ID', 'Combined Chain Key'])  # Write the header
        
        # Write each JSON filename, its corresponding PDB IDs, and combined chain keys to the CSV file
        for filename in json_filenames:
            json_file_path = os.path.join(directory_path, filename)
            pdb_ids_combined_chains = extract_pdb_ids_and_combined_chains(json_file_path)
            for pdb_id, combined_chains_keys in pdb_ids_combined_chains:
                for chain_key in combined_chains_keys:
                    writer.writerow([filename, pdb_id, chain_key])

# Example usage
directory_path = r'A:\Research\Cheng lab\PDB\propedia v2.3\json'  # Replace with your directory path
csv_file_path = r'A:\Research\Cheng lab\PDB\propedia v2.3\annotation files\annotation_files.csv'  # Replace with your desired CSV file path


record_json_filenames_pdb_ids_and_combined_chains_to_csv(directory_path, csv_file_path)

print(f"All filenames from {directory_path} have been recorded in {csv_file_path}")
