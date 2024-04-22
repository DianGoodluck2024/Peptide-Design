from Bio.PDB import PDBList
from Bio.PDB import PDBParser, is_aa
import os
import time
##User Initialization
inputFilePath = 'A:\\Research\\Cheng lab\\PDB\\PDB data\\04_13_2024_allPDBEntries_part2.txt'
failListFilePath='A:\\Research\\Cheng lab\\PDB\\PDB data\\04_13_2024_failList_part2.txt'
parentPath='A:\\Research\\Cheng lab\\PDB\\PDB data'
output_fileName = '04_13_2024_allPDBcomplexes_part2.txt'

#function
def analyze_protein_peptide_complex(pdb_file_path):
    # Initialize PDBParser
    parser = PDBParser()
    structure = parser.get_structure("PDB_structure", pdb_file_path)

    # Define the criterion for peptide vs protein
    peptide_max_length = 50

    # Initialize dictionary to store comprehensive molecule information
    molecule_info = {}
    protein_count = 0
    peptide_count = 0

    # Read PDB file and extract molecule information
    compnd = structure.header.get('compound', {})

    for key, value in compnd.items():
    # Attempt to extract molecule name and chain identifiers
        molecule_name = value['molecule']
        molecule_info[molecule_name] = {'Chains': [], 'Total AA': 0, 'Type': '', 'Chain Length':{}}
        chain_ids = value.get('chain')
        chains = chain_ids.split(',')
        chain_list = [chain.strip().upper() for chain in chains]
        molecule_info[molecule_name]['Chains'].extend(chain_list)

    for molecule, info in molecule_info.items():
        for chain_id in info['Chains']:
            chain = structure[0][chain_id]
            aa_count=0
            for residue in chain.get_residues():
                if residue.get_resname() != "HOH":  # Check if the residue is not water
                    aa_count += 1
            info['Total AA'] += aa_count
            info['Chain Length'][chain_id] = aa_count  # Store the residue count for each chain
        info['Type'] = "Protein" if info['Total AA'] > peptide_max_length else "Peptide"
        if info['Type'] == "Protein":
            protein_count += 1
        else:
            peptide_count += 1
    return protein_count, peptide_count
def count_lines(filepath):
    line_count = 0
    with open(filepath, 'r') as file:
        for line in file:
            line_count += 1
    return line_count
def check_file_exist(file_path, timeout=120):
    """Wait for a specific file to appear within a directory for a set amount of time."""
    start_time = time.time()  # Get the current time to measure the timeout

    while True:
        if os.path.exists(file_path):  # Check if the file exists
            break  # Exit the loop and continue with the rest of the program
        elif time.time() - start_time > timeout:  # Check if the timeout has been reached
            break  # Exit the loop and continue with the rest of the program
        time.sleep(0.1)  # Wait for 0.1 second before checking again
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"fail to download")

## main
pdbl = PDBList()
output_filePath=parentPath+'\\'+output_fileName
with open(output_filePath,'w') as outputfile:
    # Read the file line by line
    with open(inputFilePath, 'r') as inputfile:
        with open(failListFilePath,'w') as failListFilePath:
            numLines=count_lines(inputFilePath)
            currentLine=1
            numFail=0
            for line in inputfile:
                pdb_id = line.strip()
                if pdb_id:  # Check if the line is not empty
                    flag=True
                    if flag:
                        try:
                            # Retrieve the PDB file and save it in the specified directory
                            pdb_file_path=pdbl.retrieve_pdb_file(pdb_id, pdir=parentPath, file_format='pdb')
                        except Exception as e:
                            failListFilePath.write(f"{pdb_id}: fail to find in server\n")
                            numFail=numFail+1
                            flag=False
                    if flag:
                        try:
                            check_file_exist(pdb_file_path)
                        except Exception as e:
                            failListFilePath.write(f"{pdb_id}: fail to download from server\n")
                            numFail=numFail+1        
                            flag=False
                    if flag:
                        try:
                            protein_count, peptide_count=analyze_protein_peptide_complex(pdb_file_path)
                            if protein_count+peptide_count>1:
                                outputfile.write(f"{pdb_id}: Proteins = {protein_count}, Peptides = {peptide_count}\n")
                            os.remove(pdb_file_path)
                        except Exception as e:
                            failListFilePath.write(f"{pdb_id}: fail to analyze\n")
                            os.remove(pdb_file_path)
                            numFail=numFail+1
                            flag=False
                print(f'Complete {currentLine} out of {numLines}.\n')
                currentLine=currentLine+1
print(f'total fail: {numFail}')