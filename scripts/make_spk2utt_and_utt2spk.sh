#!/usr/bin/env bash
#
# make_spk2utt_and_utt2spk.sh
#
# Usage:
#   ./make_spk2utt_and_utt2spk.sh <wav-directory> [output-dir]
#
#   <wav-directory>   Directory containing speaker subdirectories (e.g. id0001/, id0002/, ...)
#   [output-dir]      Optional directory where outputs will be written (default: <wav-directory>)

if [ -z "$1" ]; then
    echo "Usage: $0 <wav-directory> [output-dir]"
    exit 1
fi

wav_dir="$1"
outdir="${2:-$wav_dir}"

mkdir -p "$outdir"

spk2utt_file="$outdir/spk2utt"
utt2spk_file="$outdir/utt2spk"

> "$spk2utt_file"
> "$utt2spk_file"

# open utt2spk file once using fd 3
exec 3> "$utt2spk_file"

speakers=( "$wav_dir"/*/ )
num_speakers=${#speakers[@]}

echo "Found $num_speakers speakers in $wav_dir."
echo "Writing spk2utt and utt2spk into $outdir"

total_utts=0
processed=0

draw_bar() {
  progress=$1
  total=$2
  width=40
  filled=$(( progress*width / total ))
  echo -ne "\r["
  for ((i=0;i<filled;i++)); do echo -n "#"; done
  for ((i=filled;i<width;i++)); do echo -n "-"; done
  echo -n "] $progress/$total"
}

for spk_path in "${speakers[@]}"; do
    spk=$(basename "$spk_path")
    files=()

    while IFS= read -r -d '' f; do
        rel="${f#$wav_dir/}"
        files+=( "$rel" )
        # fast write using fd 3
        echo "$rel $spk" >&3
        total_utts=$((total_utts+1))
    done < <(find "$spk_path" -type f -name "*.wav" -print0)

    if [ ${#files[@]} -gt 0 ]; then
        echo "$spk ${files[*]}" >> "$spk2utt_file"
    fi

    processed=$((processed+1))
    draw_bar "$processed" "$num_speakers"
done

# close fd 3
exec 3>&-

echo
echo "Done! $num_speakers speakers, $total_utts utterances."
echo "  spk2utt: $spk2utt_file"
echo "  utt2spk: $utt2spk_file"

