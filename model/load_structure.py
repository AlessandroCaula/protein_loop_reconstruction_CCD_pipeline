from Bio.PDB import MMCIFParser

def load_structure(protein_path):
    protein_name_split = protein_path.split('/')
    protein_name = protein_name_split[len(protein_name_split) - 1].split('.')[0].upper()

    # Parse the PDB/mmCIF file with the MMCIFParser from the Biopython library.
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(protein_name, protein_path)
    cif_dict = parser._mmcif_dict

    return structure, cif_dict # protein_name