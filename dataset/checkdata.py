from pathlib import Path
dataset_path = Path("prepared_dataset")
for split in ["train", "validation", "test"]:
    split_path = dataset_path / split
    total = 0
    print("\n", split.upper())
    print("-" * 40)
    for folder in sorted(split_path.iterdir()):
        if folder.is_dir():
            count = sum(1 for f in folder.iterdir() if f.is_file())
            total += count
            print(folder.name, ":", count)
    print("Total images:", total)
