from Bio.PDB import PDBParser, NeighborSearch

# Load the structure
parser = PDBParser()
structure = parser.get_structure("Protein_Peptide", r"A:\Research\Cheng lab\PDB\PDB data\protein-peptide complex\2an6.pdb")

# Get atoms from the structure
atoms = [atom for atom in structure.get_atoms()]

# Create NeighborSearch object
ns = NeighborSearch(atoms)

# Define the peptide atoms
peptide_chain_id = 'E'  # Peptide chain ID
peptide_atoms = [atom for atom in structure[0][peptide_chain_id].get_atoms()]

# Find all protein atoms within 4Å of the peptide atoms and exclude peptide's own atoms
interacting_residues = set()
for peptide_atom in peptide_atoms:
    nearby_atoms = ns.search(peptide_atom.coord, 4.0, level='R')  # 'R' for Residue level
    for nearby_residue in nearby_atoms:
        # Exclude residues from the same peptide chain
        if nearby_residue.get_full_id()[2] != peptide_chain_id:
            interacting_residues.add(nearby_residue)

# Print out interacting residue information along with their position index
for residue in sorted(interacting_residues, key=lambda x: x.get_full_id()):
    chain_id = residue.get_full_id()[2]
    resname = residue.get_resname()  # Get the residue name (e.g., ALA)
    resseq = residue.get_id()[1]  # Get the residue sequence number
    print(f"Residue {resname} in chain {chain_id} at position {resseq} is interacting with peptide chain {peptide_chain_id}.")
