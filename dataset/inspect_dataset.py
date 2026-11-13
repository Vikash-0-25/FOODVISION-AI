from pathlib import Path
dataset_path = Path(r"C:\Users\HP\Downloads\fooddata")
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
print("FOOD DATASET IMAGE COUNTS")
print("-------------------------")
for folder in dataset_path.rglob("*"):
    if folder.is_dir() and folder.name.lower() in ["fresh", "rotten"]:
        images = [
            file for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in image_extensions
        ]
        print(f"{folder.parent.name} - {folder.name}: {len(images)} images")
print("\nDataset checking completed!")
