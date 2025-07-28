from Bio.PDB import MMCIFIO
import csv
import os

def write_metadata_file(metadata, output_dir_path, structure_name):
    """
    Write loop modeling metadata (e.g., RMSD, clashes) to a CSV file.
    """
    if not metadata:
        print("No metadata to write.")
        return

    filename = f'{output_dir_path}/{structure_name}_loop_modelling_metadata.csv'

    fieldnames = list(metadata[0].keys())
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in metadata:
            writer.writerow(entry)
    print(f"Metadata written to {filename}/loop_modelling_metadata.csv")

def save_output(output_dir, structure_name, output_structures, metadata):
    # Create the output directory id it doesn't exists
    output_dir_path = f'{output_dir}/loop_reconstruction_output'
    os.makedirs(output_dir_path, exist_ok=True)
    # Write out ensemble structures in mmCIF
    io = MMCIFIO()
    for idx, struct in enumerate(output_structures):
        io.set_structure(struct)
        io.save(f'{output_dir_path}/modeled_{structure_name}_{idx}.cif')
    if metadata:
        # Write metadata
        write_metadata_file(metadata, output_dir_path, structure_name)