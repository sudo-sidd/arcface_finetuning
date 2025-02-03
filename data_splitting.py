import os
import random
import shutil


def split_dataset(full_dataset_path, dest_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    # Set a random seed for reproducibility
    random.seed(seed)

    # Create destination directories for train, val, and test sets
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dest_path, split)
        os.makedirs(split_dir, exist_ok=True)

    # Loop over each class folder in the full dataset
    for class_name in os.listdir(full_dataset_path):
        class_dir = os.path.join(full_dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue  # Skip files if any

        # List all image files (you can customize the extensions as needed)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Shuffle the list to randomize the split
        random.shuffle(images)

        # Compute split indices
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        # Create subdirectories for each class in each split folder
        for split, split_images in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
            split_class_dir = os.path.join(dest_path, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for img_name in split_images:
                src = os.path.join(class_dir, img_name)
                dst = os.path.join(split_class_dir, img_name)
                shutil.copy(src, dst)  # You can change to shutil.move if desired

        print(
            f"Processed class '{class_name}': {len(train_images)} train, {len(val_images)} val, {len(test_images)} test images.")


# Example usage:
full_dataset_path = 'datasets/full_dataset'
dest_path = 'datasets/split_datasets/'
split_dataset(full_dataset_path, dest_path)
