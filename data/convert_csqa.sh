#!/bin/bash

set -e

input_dir=csqa_split
out_dir=csqa_quoted

mkdir -p $out_dir

for file in $input_dir/*.json; do
    out_file=$out_dir/$(basename $file)
    echo "Processing $file -> $out_file"
    python add_dividers.py --separator-type quotes $file $out_file &
done
wait
