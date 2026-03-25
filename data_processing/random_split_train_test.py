import json
import random
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Randomly split a dict JSON into train/test.')
    parser.add_argument('--input', required=True, help='Path to input JSON file')
    parser.add_argument('--output_dir', required=True, help='Directory to save output JSON files')
    args = parser.parse_args()

    with open(args.input) as f:
        mixkit_filtered = json.load(f)

    keys = list(mixkit_filtered.keys())
    random.seed(42)
    random.shuffle(keys)
    split_idx = int(0.9 * len(keys))
    train_keys = keys[:split_idx]
    test_keys = keys[split_idx:]

    print(f'Train size: {len(train_keys)}, Test size: {len(test_keys)}')

    train_dict = {k: mixkit_filtered[k] for k in train_keys}
    test_dict = {k: mixkit_filtered[k] for k in test_keys}

    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.input))[0]
    train_path = os.path.join(args.output_dir, f'{base_filename}_train.json')
    test_path = os.path.join(args.output_dir, f'{base_filename}_test.json')

    with open(train_path, 'w') as f:
        json.dump(train_dict, f)
    with open(test_path, 'w') as f:
        json.dump(test_dict, f)


if __name__ == '__main__':
    main()
