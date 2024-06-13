import argparse
import os
import json
from tqdm import tqdm


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--input_dir', help='Directory containing json files.', required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    all_json_names = os.listdir(args.input_dir)

    all_contents_list = []
    for json_name in tqdm(all_json_names):
        contents = json.load(open(os.path.join(args.input_dir, json_name), 'r'))
        all_contents_list.append(contents)

    with open(f"{args.input_dir}.json", 'w') as f:
        json.dump(all_contents_list, f, indent=2)
