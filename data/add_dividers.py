import argparse
import set_based_prompting
import json
import tqdm
def main() ->None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("output", type=str, help="Output file")
    parser.add_argument("--separator-type", choices=["quotes", "brackets"], default="quotes", help="Separator type")
    args = parser.parse_args()
    with open(args.input, "rt") as f:
        dat = json.load(f)
    new_dat = []
    for prompt in tqdm.tqdm(dat, desc="Adding dividers"):
        new_dat.append({
            "prompt": add_dividers(prompt["prompt"], args.separator_type),
            'prompt_metadata': prompt.get("prompt_metadata", None),
        })

    with open(args.output, "wt") as f:
        json.dump(new_dat, f, indent=2)

def add_dividers(prompt: str, separator_type: str) -> str:
    split_org =set_based_prompting.SplitPrompt(text=prompt, metadata=None)

    prefix, parallel_substrings, suffix = split_org.gen_split_text()
    new_parallel_substrings = []
    for substring in parallel_substrings:
        if separator_type == "quotes":
            moded_substring = substring.replace('"', "'")
            new_parallel_substrings.append(f'"{moded_substring}" ')
            if '"' in substring:
                print(f"Quote found in substring: {substring}")
        elif separator_type == "brackets":
            new_parallel_substrings.append(f'[{substring}] ')
            if '[' in substring or ']' in substring:
                raise ValueError(f"Bracket found in substring: {substring}")
        else:
            raise ValueError(f"Unknown separator type: {separator_type}")
    split_new = set_based_prompting.SplitPrompt.from_split_text(
        prefix, new_parallel_substrings, suffix.lstrip()
    )
    return split_new.text

if __name__ == "__main__":
    main()
