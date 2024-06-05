import json
import argparse
import os
import os.path

def main() ->None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, help="input file", default="csqa_input.json")
    parser.add_argument("--out_dir", type=str, help="output directory", default="csqa_split")
    parser.add_argument("--per_file", type=int, help="number of prompts per file", default=500)

    args = parser.parse_args()


    with open(args.in_file, "rt") as fin:
            prompts = json.load(fin)

    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(0, len(prompts), args.per_file):
        out_file = os.path.join(args.out_dir, f"csqa_input_{i:05.0f}.json")
        with open(out_file, "wt") as fout:
            json.dump(prompts[i:i+args.per_file], fout, indent=2)


if __name__ == "__main__":
    main()
