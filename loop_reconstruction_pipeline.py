import argparse
from model.load_structure import load_structure
from model.find_gaps import find_gaps
from model.model_missing_loops import model_missing_loops
from model.save_output import save_output

def main():
    parser = argparse.ArgumentParser(description="Loop reconstruction pipeline")

    parser.add_argument(
        "structure", 
        help="Path to PDB/mmCIF file"
    )
    parser.add_argument(
        "--output_dir",
        default="./",
        help="Directory to save output files (default: current directory)"
    )
    parser.add_argument(
        "--n_models",
        type=int,
        default="3",
        help="Number of new generated models (ensemble) (default: 5)"
    )

    args = parser.parse_args()
    # Load the PDB/mmCIF structure
    structure, cif_dict = load_structure(args.structure)
    # Parse the structure and identify the internal missing gaps
    missing_loops = find_gaps(structure, cif_dict)
    # Method for 3D structure reconstruction
    # output_structures, metadata = model_missing_loops(structure, missing_loops, args.n_models)
    output_structures = model_missing_loops(structure, missing_loops, args.n_models)

    # Saving the 3D built structures as well as the metadata file
    structure_name = cif_dict["_struct.entry_id"][0]
    save_output(args.output_dir, structure_name, output_structures, metadata=None)

if __name__ == "__main__":
    main() 