from Bio.PDB import PDBList
from datetime import datetime

#Initialization: require modification
filePath = 'A:/Research/Cheng lab/PDB/PDB data/'
fileName = 'allPDBEntries.txt'

# Function to fetch all PDB entries
def fetch_all_pdb_entries():
    # Initialize PDBList object
    pdb_list = PDBList()
    # Obtain the current list of PDB entries
    pdb_entries = pdb_list.get_all_entries()
    return pdb_entries

# Function to write PDB entries to a text file
def write_entries_to_file(entries, filename,dateTime):
    # Get the current date and time
    current_datetime = datetime.now()
    # Format the datetime object as "month_date_year_" (e.g., "04_13_2024_")
    dateTime = current_datetime.strftime("%m_%d_%Y_")
    full_file_path = filePath + dateTime + filename  # Use full file path here
    """Write PDB entries to a text file."""
    with open(full_file_path, 'w') as file:
        for entry in entries:
            file.write(entry + '\n')

# Call the function to fetch PDB entries
pdb_entries = fetch_all_pdb_entries()

# Write entries to a text file
dateTime=''
write_entries_to_file(pdb_entries, fileName,dateTime)

print("Complete")
