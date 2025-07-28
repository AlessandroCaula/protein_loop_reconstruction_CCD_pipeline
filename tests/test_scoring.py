import unittest
import numpy as np
from Bio.PDB import Structure, Model, Chain, Residue, Atom
from model import scoring

class TestScoring(unittest.TestCase):
    def test_compute_anchor_rmsd(self):
        # Simple test: identical points should give RMSD 0
        C_end = np.array([1.0, 2.0, 3.0])
        anchor_C = np.array([1.0, 2.0, 3.0])
        N_start = np.array([4.0, 5.0, 6.0])
        anchor_N = np.array([4.0, 5.0, 6.0])
        rmsd = scoring.compute_anchor_rmsd(C_end, anchor_C, N_start, anchor_N)
        self.assertAlmostEqual(rmsd, 0.0)

        # Test with known distance
        C_end = np.array([0.0, 0.0, 0.0])
        anchor_C = np.array([3.0, 0.0, 0.0])
        N_start = np.array([0.0, 4.0, 0.0])
        anchor_N = np.array([0.0, 0.0, 0.0])
        rmsd = scoring.compute_anchor_rmsd(C_end, anchor_C, N_start, anchor_N)
        expected = np.sqrt(((3.0**2) + (4.0**2)) / 2)
        self.assertAlmostEqual(rmsd, expected)

        # Test with non-identical points
        C_end = np.array([1.0, 2.0, 3.0])
        anchor_C = np.array([2.0, 3.0, 4.0])
        N_start = np.array([4.0, 5.0, 6.0])
        anchor_N = np.array([5.0, 6.0, 7.0])
        rmsd = scoring.compute_anchor_rmsd(C_end, anchor_C, N_start, anchor_N)
        expected = np.sqrt(((1.0**2) + (1.0**2) + (1.0**2) + (1.0**2) + (1.0**2) + (1.0**2)) / 2)
        self.assertAlmostEqual(rmsd, expected)

        # Test with points having different magnitudes
        C_end = np.array([100.0, 200.0, 300.0])
        anchor_C = np.array([100.0, 200.0, 300.0])
        N_start = np.array([400.0, 500.0, 600.0])
        anchor_N = np.array([400.0, 500.0, 600.0])
        rmsd = scoring.compute_anchor_rmsd(C_end, anchor_C, N_start, anchor_N)
        self.assertAlmostEqual(rmsd, 0.0)

    def test_count_ramachandran_outliers(self):
        # All values in allowed region
        phi_psi_list = [(-60, 0), (-120, 80)]
        self.assertEqual(scoring.count_ramachandran_outliers(phi_psi_list), 0)

        # Outlier value
        phi_psi_list = [(200, 0), (-60, 200)]
        self.assertGreater(scoring.count_ramachandran_outliers(phi_psi_list), 0)

        # List with some outliers and some within the allowed regions
        phi_psi_list = [(-60, 0), (-120, 80), (200, 0), (-60, 200)]
        self.assertGreater(scoring.count_ramachandran_outliers(phi_psi_list), 0)

        # Empty list
        phi_psi_list = []
        self.assertEqual(scoring.count_ramachandran_outliers(phi_psi_list), 0)

        # List with only one φ/ψ pair
        phi_psi_list = [(-60, 0)]
        self.assertEqual(scoring.count_ramachandran_outliers(phi_psi_list), 0)

    def test_count_clashes(self):
        # Create a minimal structure with one atom at (0,0,0)
        struct = Structure.Structure("X")
        model = Model.Model(0)
        chain = Chain.Chain("A")
        res = Residue.Residue((" ", 1, " "), "GLY", "")
        atom = Atom.Atom("CA", np.array([0.0, 0.0, 0.0]), 1.0, 1.0, " ", "CA", 1, "C")
        res.add(atom)
        chain.add(res)
        model.add(chain)
        struct.add(model)

if __name__ == '__main__':
    unittest.main()

# How to run:
# python -m unittest test_scoring.py