from Bio.PDB import Atom, Residue
from Bio.Data.IUPACData import protein_letters_1to3
from .scoring import count_clashes, count_ramachandran_outliers, compute_anchor_rmsd
from .plotting import plot_backbone, plot_anchor_and_backbone
import random
import numpy as np

def normalize(v):
    """
    Normalize a 3D vector.

    Parameters:
    - v: np.ndarray of shape (3,), the input vector that needs to be normalized

    Returns:
    - np.ndarray of shape (3,), the unit vector in the same direction as v

    Raises:
    - ValueError: if the input vector has zero length
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return v / norm

def rotation_matrix(axis, theta):
    """
    Create a 3D rotation matrix to rotate a vector around a given axis by angle theta (in radians).
    Uses the quaternion-based formula to generate the rotation matrix. 

    Parameters:
    - axis: The axis to rotate around (3D vector)
    - theta: The rotation angle in radians

    Returns:
    - 3x3 rotation matrix as a NumPy array
    """

    # np.linalg.norm(x) => return the square root of the sum of the squares of the elements. For a vector [a, b, c] => sqrt(a^2 + b^2 + c^2)
    axis = normalize(axis)                  # Normalize the rotation axis vector
    a = np.cos(theta / 2.0)                 # Quaternion scalar part
    b, c, d = -axis * np.sin(theta / 2.0)   # Quaternion vector part (negated axis scaled by sin(theta/2))

    # Construct the rotation matrix from quaternion component
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d),     2*(b*d + a*c)],
        [2*(b*c + a*d),         a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c),         2*(c*d + a*b),     a*a + d*d - b*b - c*c]
    ])

# Simplified Ramachandran distributions (φ, ψ) for each amino acid
RAMACHANDRAN_DISTRIBUTIONS = {
    'ALA': {'phi': (-60, 20), 'psi': (-45, 20)},
    'ARG': {'phi': (-70, 20), 'psi': (-40, 20)},
    'ASN': {'phi': (-60, 25), 'psi': (-40, 25)},
    'ASP': {'phi': (-60, 25), 'psi': (-30, 25)},
    'CYS': {'phi': (-60, 20), 'psi': (-50, 20)},
    'GLN': {'phi': (-70, 20), 'psi': (-40, 20)},
    'GLU': {'phi': (-70, 20), 'psi': (-40, 20)},
    'GLY': {'phi': (-80, 35), 'psi': (150, 35)},   # Broad distribution
    'HIS': {'phi': (-65, 20), 'psi': (-40, 20)},
    'ILE': {'phi': (-60, 20), 'psi': (-35, 20)},
    'LEU': {'phi': (-60, 20), 'psi': (-40, 20)},
    'LYS': {'phi': (-70, 20), 'psi': (-40, 20)},
    'MET': {'phi': (-60, 20), 'psi': (-40, 20)},
    'PHE': {'phi': (-60, 20), 'psi': (-45, 20)},
    'PRO': {'phi': (-65, 10), 'psi': (135, 15)},   # Restricted φ
    'SER': {'phi': (-60, 25), 'psi': (-35, 25)},
    'THR': {'phi': (-60, 25), 'psi': (-35, 25)},
    'TRP': {'phi': (-65, 20), 'psi': (-45, 20)},
    'TYR': {'phi': (-65, 20), 'psi': (-45, 20)},
    'VAL': {'phi': (-60, 20), 'psi': (-30, 20)},
    'default': {'phi': (-60, 30), 'psi': (-40, 30)},  # Safe fallback
}

def sample_ramachandran_angles(residue_name):
    """
    Sample phi and psi angles from a simplified Ramachandran distribution.

    Parameters:
    - Residue_name: 3-letter code of the amino acid

    Returns:
    -phi, psi: sampled angles in radians
    """
    dist = RAMACHANDRAN_DISTRIBUTIONS.get(residue_name.upper(), RAMACHANDRAN_DISTRIBUTIONS['default'])
    phi = np.radians(random.gauss(dist['phi'][0], dist['phi'][1]))
    psi = np.radians(random.gauss(dist['psi'][0], dist['psi'][1]))
    return phi, psi


def build_initial_backbone(sequence, C_prev, CA_prev):
    """
    Builds an initial backbone for a loop using sampled Ramachandran phi/psi angles
    """

    # Bond lengths
    N_CA = 1.458
    CA_C = 1.525
    C_N = 1.329
    CA_C_N_ANGLE = 116.2    # Approximate angle between CA-C-N
    C_N_CA_ANGLE = 121.7    # Approximate angle between C-N-CA

    backbone = []

    # Starting direction
    direction = normalize(C_prev - CA_prev)
    N_pos = C_prev + direction * C_N
    CA_pos = N_pos + direction * N_CA
    C_pos = CA_pos + direction * CA_C
    backbone.append({'residue': sequence[0], 'N': N_pos, 'CA': CA_pos, 'C': C_pos})

    prev_N = N_pos
    prev_CA = CA_pos
    prev_C = C_pos

    for i, res_name in enumerate(sequence[1:], start=1):
        phi, psi = sample_ramachandran_angles(res_name)

        # Set N position using prev_C and rotation
        N_dir = normalize(prev_C - prev_CA)
        N_pos = prev_C + N_dir * C_N

        # Rotate from N to CA
        axis = normalize(np.cross(prev_CA - prev_N, N_dir))
        rot_CA = np.dot(rotation_matrix(axis, np.radians(C_N_CA_ANGLE)), -N_dir)
        CA_pos = N_pos + normalize(rot_CA) * N_CA

        # Rotate from CA to C using phi
        axis2 = normalize(np.cross(N_pos - prev_C, CA_pos - N_pos))
        rot_C = np.dot(rotation_matrix(axis2, phi), normalize(CA_pos - N_pos))
        C_pos = CA_pos + normalize(rot_C) * CA_C

        backbone.append({'residue': res_name, 'N': N_pos, 'CA': CA_pos, 'C': C_pos})

        prev_N = N_pos
        prev_CA = CA_pos
        prev_C = C_pos
    
    return backbone


def build_initial_backbone_OLD(sequence, C_prev, CA_prev):
    """
    Builds an initial extended backbone structure for a loop, given an amino acid sequence. 
    Each residue's backbone atoms (N, CA, C) are placed sequentially in 3D space.

    Parameters:
    - sequence: list of amino acid 3-letter codes representing the loop sequence
    - C_prev: numpy array of the position of the last known C atom before the loop
    - CA_prev: numpy array of the position of the last known CA atom before the loop

    Returns:
    - Backbone: a list of dictionaries, each containing:
        - 'residue': residue name
        - 'N': numpy array coordinate of the Nitrogen atom
        - 'CA': numpy array coordinate of the alpha Carbon atom
        - 'C': numpy array coordinate of the Carbon atom
    """

    # Typical bond lengths in Angstrom for peptide backbone atoms
    N_CA = 1.458    # Distance between N and CA atoms
    CA_C = 1.525    # Distance between CA and C atoms
    C_N = 1.329     # Distance between C and next residue's N atom

    # Initial direction vector along which atoms will be placed 
    # Start placing atoms along the x-axis by default.
    direction = normalize(C_prev - CA_prev)

    backbone = [] # Store reconstructed backbone atoms
    prev_C = C_prev # Starting point is the C atom of the anchor residue

    # Iterate over each amino acid in the input sequence
    for aa in sequence:
        # Calculate position of backbone atoms for current residue sequentially
        # N atom: located C_N distance from previous residue's C along current direction
        N = prev_C + direction * C_N

        # CA atom: located N_CA distance from N atom along same direction
        CA = N + direction * N_CA

        # C atom: located CA_C distance from CA atom along same direction
        C = CA + direction * CA_C

        # Append the coordinates and residue info in a dictionary to the backbone list
        backbone.append({'residue': aa, 'N': N, 'CA': CA, 'C': C})

        # Slightly bend the backbone by rotating the direction vector around Z axis by 30 degrees
        # This introduce a curved shape rather than a perfectly straight chain
        direction = np.dot(rotation_matrix(np.array([0, 0, 1]), np.radians(30)), direction)

        # Update prev_C to current C atom for the next residue placement
        prev_C = C

    return backbone


def rotate_torsion(backbone, residue_index, angle_rad, axis_type='phi'):
    """
    Rotates the specified torsion angle at a given residue by applying a 3D rotation to all downstream atoms.

    Parameters:
    - backbone: list of dicts with keys 'N', 'CA', 'C' and NumPy 3D coordinates
    - residue_index: index of the residue where the torsion is being rotated
    - angle_rad: rotation angle in radians
    - axis_type: 'phi' (rotation around N-CA) or 'psi' (rotation around CA-C)

    Notes:
    - For phi, atoms from the current residue and onward are rotated.
    - For psi, atoms from the next residue and onward are rotated.
    """

    # Step 1: Validate axis type and set up axis points
    if axis_type == 'phi':
        if residue_index == 0:
            return  # First residue has no φ torsion
        atom1 = backbone[residue_index]['N']
        atom2 = backbone[residue_index]['CA']
        rotate_from = residue_index     # Rotate current and forward
    elif axis_type == 'psi':
        if residue_index >= len(backbone) - 1:
            return  # Last residue has no ψ torsion
        atom1 = backbone[residue_index]['CA']
        atom2 = backbone[residue_index]['C']
        rotate_from = residue_index + 1     # Rotate next and forward
    else:
        raise ValueError("axis_type must be either 'phi' or 'psi'")

    # Step 2: Define rotation axis
    axis_origin = atom1
    axis_vector = normalize(atom2 - atom1)
    rot_matrix = rotation_matrix(axis_vector, angle_rad)

    # Step 3: Apply rotation to all downstream atoms
    for i in range(rotate_from, len(backbone)):
        for atom_key in ['N', 'CA', 'C']:
            # Translate atom to origin of rotation axis
            atom_vector = backbone[i][atom_key] - axis_origin
            # Rotate around axis
            rotate_vector = np.dot(rot_matrix, atom_vector)
            # Translate back to global coordinates
            backbone[i][atom_key] = rotate_vector + axis_origin

def vector_angle_and_axis(a, b):
    """
    Compute the angle and rotation axis needed to rotate vector 'a' to align it with vector 'b'.
    Used in CCD to determine how much a torsion should be rotated to bring the end closer to the target
    """
    a_norm = normalize(a)   # Normalize vector a (convert it to unit length)
    b_norm = normalize(b)   # Normalize vector b

    axis = np.cross(a_norm, b_norm)     # Compute axis perpendicular to both a and b (rotation axis)
    norm_axis = np.linalg.norm(axis)    # Compute length of that axis vector

    if norm_axis < 1e-6:
        # If cross product is near zero, a and b are either parallel or opposite.
        # No rotation needed (or rotate around any axis), we return a dummy axis.
        return 0.0, np.array([1.0, 0.0, 0.0])
    
    axis = axis / norm_axis     # Normalize the rotation axis
    dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)    # Clamp dot product to avoid numerical issues
    angle = np.arccos(dot)  # Compute angle between a and b using arccos (dot product)
    return angle, axis  # Return angle and normalized axis of rotation


def ccd_close_loop(backbone, target_CA, max_iterations=100, tolerance=0.8):
    """
    Applies the CCD (Cyclic Coordinate Descent) algorithm to iteratively modify the φ and ψ angles
    of a protein loop to bring the end (last CA atom in the loop) closer to the anchor (target_CA)
    """

    for iteration in range(max_iterations):
        # Get position of current final CA in the loop
        current_end = backbone[-1]['CA']

        # Compute Euclidean distance to the target anchor CA (Closure error)
        current_error = np.linalg.norm(target_CA - current_end)

        # If the end is close enough to the anchor, exit successfully
        if current_error < tolerance:
            print(f"CCD converged in {iteration} iterations.")
            return backbone, True
        
        # Go though each residue in the loop (excluding fixed ends)
        for i in range(1, len(backbone) - 1):
            # For each residue, try rotating both φ and ψ torsion angles.
            for axis_type in ['phi', 'psi']:

                # Skip φ rotation for the first residue (no φ before it)
                if axis_type == 'phi' and i == 0:
                    continue
                # Skip ψ rotation for the last residue (no ψ after it)
                if axis_type == 'psi' and i >= len(backbone) - 1:
                    continue

                # Select the rotation axis: either φ (around CA-N) or ψ (around CA-C)
                axis_point = backbone[i]['CA']  # Both φ and ψ rotate around the CA atom
                if axis_type == 'phi':
                    end_point = backbone[i]['N']    # φ axis is CA -> N
                else:
                    end_point = backbone[i]['C']    # ψ axis is CA -> C
                
                # Vector from the axis point to current end CA of the loop.
                current_vector = backbone[-1]['CA'] - axis_point
                # Vector form the axis point to the desired target CA (anchor)
                target_vector = target_CA - axis_point

                # Compute rotation (angle and axis) to bring current_vector closer to target_vector
                angle, rotation_axis = vector_angle_and_axis(current_vector, target_vector)

                # Apply the rotation to all downstream atoms (φ or ψ rotation)
                # Scale the rotation angle (e.g., 0.5) to smooth convergence and avoid overshooting
                rotate_torsion(backbone, i, angle_rad=angle * 0.7, axis_type=axis_type)
    
    # If all iterations are used and loop is still open, report failure
    print("CCD did not converge")
    return backbone, False


def rebuild_N_CA_C(prev_C, curr_CA, next_CA):
    """
    Reconstruct the N and C atoms around a given CA atom using ideal geometry.

    Parameters:
    - prev_C (np.array): Coordinates of the previous residue's C atom
    - prev_CA (np.array): Coordinates of the current residue's CA atom
    - next_CA (np.array): Coordinates of the next residue's CA atom

    Returns:
    - N (np.array): Reconstructed N atom coordinates
    - C (np.array): Reconstructed C atom coordinates
    """
    # Ideal bond lengths (in angstroms)
    CA_N_len = 1.458
    CA_C_len = 1.525

    # Ideal bond angles N-CA-C (in radians)
    angle_N_CA_C = np.radians(111.2)

    # --- Rebuild N 
    # Direction from CA to previous C
    dir_N = normalize(prev_C - curr_CA)
    # Orthogonal vector using the plane defined by prev_C, curr_CA, next_CA
    ortho_N = normalize(np.cross(dir_N, next_CA - curr_CA))
    # Final N position using ideal angle
    N = curr_CA + CA_N_len * (np.cos(angle_N_CA_C) * dir_N +
                              np.sin(angle_N_CA_C) * ortho_N)
    
    # --- Rebuild C 
    # Direction from CA to next CA
    dir_C = normalize(next_CA - curr_CA)
    # Orthogonal vector using the plane defined by next_CA, curr_CA, prev_C
    ortho_C = normalize(np.cross(dir_C, prev_C - curr_CA))
    # Final C position using ideal angle
    C = curr_CA + CA_C_len * (np.cos(angle_N_CA_C) * dir_C +
                              np.sin(angle_N_CA_C) * ortho_C)
    return N, C


def model_missing_loops(structure, missing_loops, n_models = 1):

    # Helper function: convert one letter code to three letter code.
    def one_to_three(one_letter_aa):
        return protein_letters_1to3.get(one_letter_aa.upper(), "UNK").upper()

    # Helper function: build a residue from backbone coordinates
    def build_residue(name, res_id, atoms_coords):
        residue_id = (" ", res_id, " ")
        new_residue = Residue.Residue(residue_id, name, "")
        # Add atoms_coords atoms
        atoms = {
            'N': atoms_coords['N'],
            'CA': atoms_coords['CA'],
            'C': atoms_coords['C']
        }
        for atom_name, coord in atoms.items():
            atom = Atom.Atom(atom_name, coord, 1.0, 1.0, ' ', atom_name, 0, element=atom_name[0])
            new_residue.add(atom)
        return new_residue

    # missing_loops = [missing_loops[0]]

    output_structures = []
    for model in structure:
        for i in range(n_models):
            new_model = model.copy()
            for loop in missing_loops:
                chain = model[loop.chain_id]
                new_chain = new_model[loop.chain_id]

                # --- Get the anchor amino acid
                anchor_C_v = chain[loop.start_res - 1]['C'].get_vector() # C atom of residue before loop
                anchor_C = np.array([c for c in anchor_C_v]) # Convert it to np array
                anchor_CA_v = chain[loop.start_res - 1]['CA'].get_vector() # CA atom of residue before loop
                anchor_CA = np.array([c for c in anchor_CA_v]) # Convert it to np array

                # Missing loop sequence
                missing_loop_sequence = [one_to_three(res) for res in loop.sequence]

                # --- Get the target anchor CA (residue after the loop)
                target_CA_v = chain[loop.end_res + 1]['CA'].get_vector()
                target_CA = np.array([c for c in target_CA_v])
                
                # --- Loop modeling
                max_attempts = 10
                for attempt in range(max_attempts):
                    # --- Building the initial backbone
                    backbone = build_initial_backbone(missing_loop_sequence, anchor_C, anchor_CA)

                    # --- Run CCD to close the loop
                    backbone, converged = ccd_close_loop(backbone, target_CA)

                    # Print CCD result
                    if converged:
                        # print(f"Loop {loop.start_res}-{loop.end_res} closed successfully.")

                        # --- Rebuild N and C atoms after CCD adjusted CA
                        for i in range(len(backbone)):
                            curr_CA = backbone[i]['CA']

                            # Get previous C atom
                            if i == 0:
                                prev_C = anchor_C
                            else:
                                prev_C = backbone[i - 1]['C']

                            # Get next CA atom
                            if i < len(backbone) - 1:
                                next_CA = backbone[i + 1]['CA']
                            else:
                                next_CA = target_CA

                            # Rebuild N and C
                            N_pos, C_pos = rebuild_N_CA_C(prev_C, curr_CA, next_CA)

                            # Store updated atoms
                            backbone[i]['N'] = N_pos
                            backbone[i]['C'] = C_pos

                        break
                #     else:
                #         print(f"Loop {loop.start_res}-{loop.end_res} failed to close.")
                # else:
                #     print(f"CCD failed to converge after {max_attempts} attempts.")

                for i, res_id in enumerate((range(loop.start_res, loop.end_res + 1))):
                    res_name = backbone[i]['residue']
                    # --- Build new loop residue to add to the final chain
                    new_residue = build_residue(res_name, res_id, backbone[i])
                    # Add new residue to chain 
                    new_chain.add(new_residue)
                
                # # --- Plot the CA backbone chain
                # plot_anchor_and_backbone(anchor_CA, target_CA, backbone)
            
            # Add the new model to the final output structure
            output_structures.append(new_model)

    return output_structures
