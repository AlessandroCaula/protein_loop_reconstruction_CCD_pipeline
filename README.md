# Loop Reconstruction Pipeline

This project provides a pipeline for identifying and modeling missing internal loops in protein structures using PDB/mmCIF files. It parses the structure, detects unresolved regions, and generates plausible backbone conformations for missing loops. The modeling step is performed using the Cyclic Coordinate Descent (CCD) algorithm, which iteratively adjusts backbone torsion angles (φ and ψ) to geometrically "close" the loop between known anchor residues, ensuring spatial continuity of the protein backbone.

## Features

- **Automatic detection of missing internal loops** in protein structures.
- **Backbone modeling** for missing loops using φ/ψ angle sampling and geometric constraints.
- **Scoring** of generated models for steric clashes, Ramachandran outliers, and anchor RMSD.
- **Output**: Modeled structures in mmCIF format and a CSV file with modeling metadata.

## Requirements

- Python 3.7+
- [Biopython](https://biopython.org/) (for structure parsing and manipulation)
- numpy
- matplotlib

Install dependencies with:

```bash
pip install biopython numpy matplotlib
```

or

```bash
pip install -r requirements.txt
```

## Usage 

```bash
python loop_reconstruction_pipeline.py <input_PDBmmCIF_file_path> --output_dir <output_path> ----n_models <number_of_generated_models>
```

## Output

- Modeled structures are saved in loop_reconstruction_output/ as modeled_0.cif, modeled_1.cif, etc.
- Modeling metadata (e.g., RMSD, clashes) is saved as loop_modelling_metadata.csv in the same folder.

## Notes

- The pipeline currently models only internal missing loops (not N- or C-terminal gaps) and backbone structure (no side-chains).
- The code is modular and can be extended for additional modeling or scoring features.
