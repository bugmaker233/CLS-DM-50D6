import argparse
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create train/test/dis txt files from <dataroot>/*/ct_xray_data.h5"
    )
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=Path("/root/data1/CTSpine1K/mid_h5"),
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default="ct_xray_data.h5",
        help="Filename to scan recursively.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--dis-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Folder to write train.txt/test.txt/dis.txt",
    )
    return parser.parse_args()


def validate_ratios(train_ratio, test_ratio, dis_ratio):
    total = train_ratio + test_ratio + dis_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"Ratios must sum to 1.0, got {total:.6f} "
            f"(train={train_ratio}, test={test_ratio}, dis={dis_ratio})."
        )
    if min(train_ratio, test_ratio, dis_ratio) < 0:
        raise ValueError("Ratios must be non-negative.")


def write_list(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.test_ratio, args.dis_ratio)

    dataroot = args.dataroot.resolve()
    if not dataroot.exists():
        raise FileNotFoundError(f"dataroot not found: {dataroot}")

    files = list(dataroot.rglob(args.target_name))
    if not files:
        raise RuntimeError(f"No '{args.target_name}' found in {dataroot}")

    case_ids = sorted(
        {
            file_path.parent.relative_to(dataroot).as_posix()
            for file_path in files
        }
    )

    random.Random(args.seed).shuffle(case_ids)

    n_total = len(case_ids)
    n_train = int(n_total * args.train_ratio)
    n_test = int(n_total * args.test_ratio)
    n_dis = n_total - n_train - n_test

    train_ids = case_ids[:n_train]
    test_ids = case_ids[n_train : n_train + n_test]
    dis_ids = case_ids[n_train + n_test :]

    output_dir = args.output_dir.resolve()
    write_list(output_dir / "train.txt", train_ids)
    write_list(output_dir / "test.txt", test_ids)
    write_list(output_dir / "dis.txt", dis_ids)

    print(f"dataroot: {dataroot}")
    print(f"matched cases: {n_total}")
    print(f"train/test/dis: {len(train_ids)}/{len(test_ids)}/{len(dis_ids)}")
    print(f"written to: {output_dir}")
    print(f"dis size check: {n_dis}")


if __name__ == "__main__":
    main()
