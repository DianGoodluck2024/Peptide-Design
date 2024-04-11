from Bio.PDB import PDBParser

# Define the path to your PDB file
pdb_file_path = 'A:\\Research\\Cheng lab\\PDB\\PDB data\\protein-peptide complex\\2an6.pdb'
#pdb_file_path = 'A:\\Research\\Cheng lab\\PDB\\PDB data\\peptide complex\\1jym.pdb'

parser = PDBParser()
structure = parser.get_structure("PDB_structure", pdb_file_path)

# Define the criterion for peptide vs protein
peptide_max_length = 50

# Initialize dictionary to store comprehensive molecule information
molecule_info = {}
protein_count = 0
peptide_count = 0

molecule_name=''

with open(pdb_file_path, 'r') as file:
    for line in file:
        if line.startswith('COMPND') and 'MOLECULE:' in line:
            molecule_name = line.split('MOLECULE:')[1].split(';')[0].strip()
            # Initialize dictionary for this molecule
            molecule_info[molecule_name] = {'Chains': [], 'Total AA': 0, 'Type': ''}
        elif line.startswith('COMPND') and 'CHAIN:' in line and molecule_name:
            chains = line.split('CHAIN:')[1].split(';')[0].strip().split(',')
            chain_list = [chain.strip() for chain in chains]
            if molecule_name:  # Make sure molecule_name is not empty
                molecule_info[molecule_name]['Chains'].extend(chain_list)
                molecule_name=''

for molecule, info in molecule_info.items():
    for chain_id in info['Chains']:
        chain = structure[0][chain_id]
        aa_count = sum(1 for residue in chain.get_residues() if residue.get_id()[0] == ' ')
        info['Total AA'] += aa_count
    info['Type'] = "Protein" if info['Total AA'] > peptide_max_length else "Peptide"
    if info['Type'] == "Protein":
        protein_count += 1
    else:
        peptide_count += 1

# Print the comprehensive molecule information
for molecule, info in molecule_info.items():
    print(f"{molecule}: Chains = {info['Chains']}, Total AA = {info['Total AA']}, Type = {info['Type']}")

print(f"Total number of proteins: {protein_count}")
print(f"Total number of peptides: {peptide_count}")