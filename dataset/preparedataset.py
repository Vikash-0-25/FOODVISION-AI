from pathlib import Path
import random
import shutil
source_dir = Path(r"C:\Users\HP\AI-Food-Freshness-Prediction\dataset\fooddata")
output_dir = Path("prepared_dataset")
image_extensions = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}
status_mapping = {
    "fresh": "Fresh",
    "highlyfresh": "Fresh",
    "rotten": "Rotten",
    "notfresh": "Rotten"
}
if not source_dir.exists():
    raise FileNotFoundError("Original dataset not found!")
if output_dir.exists():
    raise FileExistsError(
        "prepared_dataset already exists. "
        "Do not run this script again without checking it."
    )
random.seed(42)
class_images = {}
for category_folder in source_dir.iterdir():
    if not category_folder.is_dir():
        continue
    for food_folder in category_folder.iterdir():
        if not food_folder.is_dir():
            continue
        food_name = food_folder.name
        for status_folder in food_folder.iterdir():
            if not status_folder.is_dir():
                continue
            status = status_mapping.get(
                status_folder.name.lower()
            )
            if status is None:
                continue
            class_name = food_name + "_" + status
            images = [
                file for file in status_folder.iterdir()
                if file.is_file()
                and file.suffix.lower() in image_extensions
            ]
            class_images.setdefault(
                class_name, []
            ).extend(images)
print("\nORIGINAL DATASET SUMMARY")
print("-------------------------")
for class_name, images in sorted(class_images.items()):
    print(class_name, ":", len(images))
splits = ["train", "validation", "test"]
for split in splits:
    (output_dir / split).mkdir(parents=True)
print("\nSPLITTING DATASET...")
print("--------------------")
for class_name, images in sorted(class_images.items()):
    random.shuffle(images)
    total = len(images)
    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)
    split_images = {
        "train": images[:train_end],
        "validation": images[train_end:validation_end],
        "test": images[validation_end:]
    }
    for split, selected_images in split_images.items():
        destination = output_dir / split / class_name
        destination.mkdir(parents=True, exist_ok=True)
        for index, image_path in enumerate(selected_images):
            new_name = (
                str(index) + "_" + image_path.name
            )
            shutil.copy2(
                image_path,
                destination / new_name
            )
    print(
        class_name,
        "-> Train:", len(split_images["train"]),
        "Validation:", len(split_images["validation"]),
        "Test:", len(split_images["test"])
    )
print("\nDATASET PREPARATION COMPLETED!")
print("Prepared dataset location:", output_dir.resolve())
