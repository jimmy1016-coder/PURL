"""
CN-Celeb trials.lst를 VoxCeleb 형식 (label file1 file2)으로 변환.

trials.lst: enroll_id test_file label
enroll.lst: enroll_id enroll_path
→ pairs: label enroll_path test_path  (경로는 .flac로)

Usage:
  python scripts/convert_cnceleb_trials_to_pairs.py \
    --eval_dir /path/to/CN-Celeb_flac/eval \
    --output /path/to/chns/resources/cnceleb_eval_pairs.txt
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--ext", type=str, default=".flac", help="Audio extension (.flac or .wav)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    enroll_lst = eval_dir / "lists" / "enroll.lst"
    trials_lst = eval_dir / "lists" / "trials.lst"

    # enroll_id -> enroll_path (with correct extension)
    enroll_map = {}
    with open(enroll_lst) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            enroll_id, enroll_path = line.split(maxsplit=1)
            # replace .wav with actual extension
            enroll_path_flac = enroll_path.replace(".wav", args.ext)
            enroll_map[enroll_id] = enroll_path_flac

    pairs = []
    with open(trials_lst) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            enroll_id = parts[0]
            test_file = parts[1]
            label = parts[2]
            enroll_path = enroll_map[enroll_id]
            test_path = test_file.replace(".wav", args.ext)
            pairs.append((label, enroll_path, test_path))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for label, file1, file2 in pairs:
            f.write(f"{label} {file1} {file2}\n")

    print(f"Wrote {len(pairs)} pairs to {output_path}")
    print(f"  file1 = enroll paths, file2 = test paths")
    print(f"  data_dir should be: {eval_dir}")


if __name__ == "__main__":
    main()
