from Bio.PDB import is_aa # MMCIFParser
from Bio.Data.IUPACData import protein_letters_3to1
from collections import defaultdict, namedtuple

def get_observed_residues(structure):
    """
    Extracts observed residues (with coordinates) from the structure. 
    Returns a dict if {chain_id: [res_id1, res_id2, ...]]}
    """
    observed_res = defaultdict(list)
    for model in structure:
        for chain in model:
            for res in chain:
                if is_aa(res, standard=True):
                    observed_res[chain.id].append(res.id[1])
    return observed_res

def identify_internal_gaps(observed, mmCIF_dict):
    """
    Extract the missing gap residues and their position in the pdb structure
    missing_res = {chain_id: [(gap1_res1, gap1_idx1), (gap1_res2, gap1_idx2), ..], [(gap2_res1, gap2_idx1), (gap2_res2, gap2_idx2), ..], ..}
    internal_gaps = (chain_id, gap_start_idx, gap_end_idx, gap_sequence_res)
    """

    # Convert three letter code aa to one letter code aa
    def three_to_one(three_letter_aa):
        return protein_letters_3to1.get(three_letter_aa.capitalize(), "X")

    chain_id = mmCIF_dict['_pdbx_poly_seq_scheme.pdb_strand_id']        # Chain ID
    seq_res_names = mmCIF_dict['_pdbx_poly_seq_scheme.mon_id']          # Three-letter code. From the reference sequence
    auth_seq_nums = mmCIF_dict['_pdbx_poly_seq_scheme.auth_seq_num']    # Author-assigned residue number. From the protein model.
    pdb_seq_nums = mmCIF_dict['_pdbx_poly_seq_scheme.pdb_seq_num']      # Residue number used in the PDB structure.

    # sort the observed lists
    for key in observed:
        observed[key].sort()

    # Final collection. dict = {"chain_id": [(gap1_res1, gap1_idx1), (gap1_res2, gap1_idx2)]}
    missing_res = defaultdict(list)
    
    open_gap = False        # Boolean used to check whether you are in an open gap or not
    curr_missing_res = []   # Collection storing the current gap
    # Loop through all the collections of sequences, and extract the missing internal loop
    for i in range(len(chain_id)):
        if not pdb_seq_nums[i].isnumeric():
            continue
        # Retrieve the start and end res index of the resolved PDB structure
        start_pdb_idx = observed[chain_id[i]][0]
        end_pdb_idx = observed[chain_id[i]][len(observed[chain_id[i]]) - 1]
        # Check if the current residue is within the start and end residues of the resolved structure. Considering only internal missing gaps.
        if int(pdb_seq_nums[i]) >= start_pdb_idx and int(pdb_seq_nums[i]) <= end_pdb_idx:
            # If the current residue does not have a numeric index, it is instead unknown ('?', '.'), then save it in the missing gap collection. 
            if not auth_seq_nums[i].isnumeric():
                open_gap = True
                # Store the index and residue of the current missing residue gap
                curr_missing_res.append((int(pdb_seq_nums[i]), three_to_one(seq_res_names[i])))
            else:
                if open_gap:
                    # If the curr_missing_res is not empty. Add it to the final collection of missing res
                    if len(curr_missing_res) != 0:
                        missing_res[chain_id[i]].append(curr_missing_res)
                    curr_missing_res = []
                    open_gap = False    # Close the missing gap if it is open

    # Reorganize the missing residue collection
    internal_gaps = []
    for chain in missing_res.keys():
        for gap_idx in range(len(missing_res[chain])):
            Loop = namedtuple("Loop", ["chain_id", "start_res", "end_res", "sequence"])
            # Sort the collection
            sorted_res = sorted(missing_res[chain][gap_idx], key=lambda x: x[0])
            start_gap_idx = missing_res[chain][gap_idx][0][0]
            end_gap_idx = missing_res[chain][gap_idx][len(missing_res[chain][gap_idx]) - 1][0]
            # String sequence of the aa
            missing_res_seq = ''
            for res in missing_res[chain][gap_idx]:
                missing_res_seq += res[1]

            # For loop modeling and analysis, loops of less than 4 residues are often discarded due to their short length and potential difficulty in prediction.
            if len(missing_res_seq) >= 4:
                # Add everything the gaps information to the final collection
                internal_gaps.append(Loop(chain_id=chain, start_res=start_gap_idx, end_res=end_gap_idx, sequence=missing_res_seq))

    return internal_gaps

def find_gaps(structure, cif_dict):
    """
    Retrieve the resolved PDB sequence structure and identify the internal missing loop gaps
    """
    observed = get_observed_residues(structure)         # Retrieve the resolved in the PDB structure 
    gaps = identify_internal_gaps(observed, cif_dict)   # Identify internal loop gaps

    return gaps