import os
import cv2
import numpy as np
import shutil  # To move directories
from sklearn.model_selection import train_test_split

# Step 1: Convert real images to sketches
def generate_sketches(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for room_folder in os.listdir(input_folder):
        room_path = os.path.join(input_folder, room_folder)
        if os.path.isdir(room_path):
            sketch_folder = os.path.join(output_folder, f'{room_folder}_sketch')
            os.makedirs(sketch_folder, exist_ok=True)
            for filename in os.listdir(room_path):
                # Check for non-jpg files and skip them
                if not filename.endswith('.jpg'):
                    print(f"Skipping non-jpg file: {filename}")
                    continue

                image_path = os.path.join(room_path, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    inverted_image = 255 - gray_image
                    blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), 0)
                    inverted_blurred = 255 - blurred_image
                    sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

                    sketch_name = f"sketch_{filename}"
                    sketch_path = os.path.join(sketch_folder, sketch_name)
                    cv2.imwrite(sketch_path, sketch)
                    print(f"Sketch saved: {sketch_path}")
                else:
                    print(f"Failed to load image: {image_path}")

# Step 2: Prepare dataset for Pix2Pix
def prepare_pix2pix_dataset(real_folder, sketch_folder, output_folder):
    # Create output folders
    A_folder = os.path.join(output_folder, 'A')
    B_folder = os.path.join(output_folder, 'B')
    for subfolder in ['train', 'val', 'test']:  # Include an empty test folder
        os.makedirs(os.path.join(A_folder, subfolder), exist_ok=True)
        os.makedirs(os.path.join(B_folder, subfolder), exist_ok=True)

    # Collect all images
    real_images, sketch_images = [], []
    for root, _, files in os.walk(real_folder):
        real_images += [os.path.join(root, file) for file in files if file.endswith('.jpg')]
    for root, _, files in os.walk(sketch_folder):
        sketch_images += [os.path.join(root, file) for file in files if file.startswith('sketch_')]

    # Debugging: Log the counts
    print(f"Number of real images: {len(real_images)}")
    print(f"Number of sketch images: {len(sketch_images)}")

    # Match real and sketch images
    real_images = [img for img in real_images if os.path.basename(img) in {os.path.basename(sketch)[7:] for sketch in sketch_images}]
    sketch_images = [sketch for sketch in sketch_images if os.path.basename(sketch)[7:] in {os.path.basename(img) for img in real_images}]

    # Debugging: Log post-filtering counts
    print(f"Filtered real images: {len(real_images)}")
    print(f"Filtered sketch images: {len(sketch_images)}")

    # Split dataset into 90% train and 10% val
    train_real, val_real, train_sketch, val_sketch = train_test_split(real_images, sketch_images, test_size=0.1, random_state=42)

    # Dataset splits
    datasets = {
        'train': (train_real, train_sketch),
        'val': (val_real, val_sketch),
        'test': ([], [])  # Leave the test folder empty
    }

    # Copy and rename files
    for split, (real_set, sketch_set) in datasets.items():
        for idx, (real, sketch) in enumerate(zip(real_set, sketch_set), start=1):
            real_dest = os.path.join(B_folder, split, f"{idx}.jpg")
            sketch_dest = os.path.join(A_folder, split, f"{idx}.jpg")
            shutil.copy(real, real_dest)
            shutil.copy(sketch, sketch_dest)
            print(f"Copied: {real} -> {real_dest}, {sketch} -> {sketch_dest}")

# Step 3: Combine A and B for Pix2Pix
def combine_A_and_B(folder_A, folder_B, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def combine_images(path_A, path_B, path_AB):
        im_A = cv2.imread(path_A)
        im_B = cv2.imread(path_B)
        if im_A is not None and im_B is not None:
            im_AB = np.concatenate([im_A, im_B], axis=1)
            cv2.imwrite(path_AB, im_AB)

    splits = ['train', 'val', 'test']
    for split in splits:
        folder_split_A = os.path.join(folder_A, split)
        folder_split_B = os.path.join(folder_B, split)
        folder_split_AB = os.path.join(output_folder, split)
        os.makedirs(folder_split_AB, exist_ok=True)

        A_images = sorted(os.listdir(folder_split_A))
        B_images = sorted(os.listdir(folder_split_B))

        for img_A, img_B in zip(A_images, B_images):
            path_A = os.path.join(folder_split_A, img_A)
            path_B = os.path.join(folder_split_B, img_B)
            path_AB = os.path.join(folder_split_AB, img_A)  # Save with the same name as A
            combine_images(path_A, path_B, path_AB)
            print(f"Combined: {path_A} + {path_B} -> {path_AB}")

# Step 4: Move dataset_AB to pytorch-CycleGAN-and-pix2pix/datasets
def move_dataset_to_pytorch_folder(dataset_folder, pytorch_folder):
    datasets_folder = os.path.join(pytorch_folder, 'datasets')
    if not os.path.exists(datasets_folder):
        os.makedirs(datasets_folder)

    destination = os.path.join(datasets_folder, os.path.basename(dataset_folder))
    if os.path.exists(destination):
        shutil.rmtree(destination)  # Remove existing folder to avoid conflicts
    shutil.move(dataset_folder, destination)
    print(f"Dataset moved to: {destination}")

# Main function to execute all tasks
if __name__ == "__main__":
    base_folder = r'C:\Users\pytorch\Desktop\pix_to_pix'
    pytorch_folder = r'C:\Users\pytorch\Desktop\pix_to_pix\pytorch-CycleGAN-and-pix2pix'
    real_images_folder = os.path.join(base_folder, 'bedroom_dataset_resized')
    sketch_images_folder = os.path.join(base_folder, 'bedroom_dataset_resized_Sketch')
    pix2pix_dataset_folder = os.path.join(base_folder, 'dataset_for_pix2pix')
    combined_dataset_folder = os.path.join(base_folder, 'dataset_AB')

    # Step 1: Generate sketches
    generate_sketches(real_images_folder, sketch_images_folder)

    # Step 2: Prepare dataset for Pix2Pix
    prepare_pix2pix_dataset(real_images_folder, sketch_images_folder, pix2pix_dataset_folder)

    # Step 3: Combine A and B folders
    combine_A_and_B(
        os.path.join(pix2pix_dataset_folder, 'A'),
        os.path.join(pix2pix_dataset_folder, 'B'),
        combined_dataset_folder
    )

    # Step 4: Move dataset_AB to pytorch-CycleGAN-and-pix2pix/datasets
    move_dataset_to_pytorch_folder(combined_dataset_folder, pytorch_folder)
