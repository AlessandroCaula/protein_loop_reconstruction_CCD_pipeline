from Bio.PDB import NeighborSearch
import numpy as np

def count_clashes(N_coords, CA_coords, C_coords, model, threshold=2.0):
    """
    Counts the number of steric clashes between loop atoms and existing atoms in the model.
    Clash = distance < threshold Å (default 2.0 Å)
    """

    all_coords = N_coords + CA_coords + C_coords
    loop_atoms = [tuple(coord) for coord in all_coords]

    # Gather all non-loop atoms in the structure
    atoms = [atom for atom in model.get_atoms() if atom.element != 'H']
    ns = NeighborSearch(atoms)

    clash_count = 0
    for coord in loop_atoms:
        neighbors = ns.search(coord, threshold)
        clash_count += len(neighbors)
    
    return clash_count

def count_ramachandran_outliers(phi_psi_list, sequence=None):
    """
    Count number of φ/ψ pairs that are highly unusual for loops.
    This version uses broad regions based on known loop statistics.
    """
    outliers = 0
    for phi, psi in phi_psi_list:
        if not (-180 <= phi <= 180 and -180 <= psi <= 180):
            outliers += 1
            continue

        # Broad loop-allowed regions based on empirical loop data
        allowed = (
            (-160 <= phi <= -30 and -100 <= psi <= 100) or   # main loop α region
            (-80 <= phi <= 0 and 100 <= psi <= 180) or       # loop β region
            (-180 <= phi <= 0 and -180 <= psi <= -30) or     # some extended loops
            (60 <= phi <= 180 and -90 <= psi <= 30)          # left-handed
        )

        if not allowed:
            outliers += 1

    return outliers

def compute_anchor_rmsd(C_end, anchor_C, N_start, anchor_N):
    """
    Compute RMSD between generated loop ends and actual anchor atoms.
    Compares:
        - last C of loop to anchor_C (start of C-terminal anchor)
        - first N of loop to anchor_N (end of N-terminal anchor)
    """
    d1 = np.linalg.norm(C_end - anchor_C)
    d2 = np.linalg.norm(N_start - anchor_N)

    return np.sqrt((d1**2 + d2**2) / 2)