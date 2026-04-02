from datasets import load_from_disk
import argparse
import datasets
from pathlib import Path

from tqdm import tqdm
import copy

def main(argv=None):
    parser = argparse.ArgumentParser(description="Minimal aggregate and flatten datasets")
    parser.add_argument("--input-ds-path", required=True, help="Paths to input datasets")
    parser.add_argument("--output-ds-path", default=None, help="Optional explicit output dataset path.")

    args = parser.parse_args(argv)
    dataset = load_from_disk(args.input_ds_path)
    print(f"Loaded dataset from {args.input_ds_path}")

    new_ds = []

    for example in tqdm(dataset):

        flat_example1 = copy.deepcopy(example)
        flat_example2 = copy.deepcopy(example)

        flat_example1.pop("traj2")
        flat_example1.pop("traj2_correctness")
        flat_example1.pop("traj2_metadata")

        flat_example2["traj1"] = flat_example2.pop("traj2")
        flat_example2["traj1_correctness"] = flat_example2.pop("traj2_correctness")
        flat_example2["traj1_metadata"] = flat_example2.pop("traj2_metadata")

        new_ds.append(flat_example1)
        new_ds.append(flat_example2)

    # Save the new dataset
    if args.output_ds_path is None:
        new_ds_name = args.input_ds_path.replace("T2", "T1_flattend_from_T2")
    else:
        new_ds_name = str(Path(args.output_ds_path))
    print(f"Saving new dataset to {new_ds_name}")

    ds_new = datasets.Dataset.from_list(new_ds)
    ds_new.save_to_disk(new_ds_name)
    # Assuming you want to save it to a path, you might want to add an argument for output path
    # For now, let's just print the length of the new dataset
    print(f"Processed dataset length: {len(ds_new)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
