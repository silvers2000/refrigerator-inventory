# src/generate_cls_yaml.py
import os, argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--processed_dir",
        default="data/processed/fruits360/Training",
        help="Path to processed Training folder"
    )
    p.add_argument(
        "--output",
        default="data/fruits360-cls.yaml",
        help="Path to write the classification YAML"
    )
    args = p.parse_args()

    classes = sorted(
        d for d in os.listdir(args.processed_dir)
        if os.path.isdir(os.path.join(args.processed_dir, d))
    )
    nc = len(classes)
    names = ", ".join(f'"{c}"' for c in classes)

    yaml = f"""
train: ../data/processed/fruits360/Training
val:   ../data/processed/fruits360/Validation

nc: {nc}
names: [{names}]
""".lstrip()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(yaml)
    print(f"Wrote {args.output} with {nc} classes.")

if __name__ == "__main__":
    main()
