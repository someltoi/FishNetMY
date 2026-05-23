import os
import random
import shutil
import sys

def copy_random_images_from_subfolders():
    print("--- Random Image Picker (Parent Directory Mode) ---")
    
    # --- CONFIGURATION ---
    # This should be the PARENT folder containing the species subfolders
    # Example: .../version 2/Grouper/ (which contains coiodes, tauvina, etc.)
    parent_src_folder = "/workspaces/ai-projects/Dataset/Batch image download from google images/version 2/Bass"
    
    # Destination where ALL picked images will be copied
    dst_folder = "/workspaces/ai-projects/Dataset/Malaysian maritime fish/version 2/Seabass"
    
    # Number of images to pick PER SUBFOLDER
    num_to_pick_per_folder = 388
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    # 1. Validation
    if not os.path.exists(parent_src_folder):
        print(f"Error: Parent source folder '{parent_src_folder}' does not exist.")
        return

    if num_to_pick_per_folder <= 0:
        print("Please ensure num_to_pick is greater than 0.")
        return

    # 2. Create Destination if it doesn't exist
    if not os.path.exists(dst_folder):
        try:
            os.makedirs(dst_folder)
            print(f"Created destination folder: {dst_folder}")
        except OSError as e:
            print(f"Error creating destination folder: {e}")
            return

    # 3. Iterate through all items in the parent folder
    grand_total_copied = 0
    
    # Get a list of all items in parent folder
    items = os.listdir(parent_src_folder)
    
    # Filter to only keep directories (subfolders)
    subfolders = [d for d in items if os.path.isdir(os.path.join(parent_src_folder, d))]
    
    print(f"Found {len(subfolders)} subfolders in parent directory.")
    print("-" * 30)

    for subfolder_name in subfolders:
        current_src_path = os.path.join(parent_src_folder, subfolder_name)
        print(f"Processing subfolder: {subfolder_name}...")
        
        # Scan for images in this specific subfolder
        all_files = os.listdir(current_src_path)
        
        images = [
            f for f in all_files 
            if f.lower().endswith(valid_extensions) 
            and "Zone.Identifier" not in f
        ]
        
        total_images = len(images)
        
        if total_images == 0:
            print(f"  -> No images found in {subfolder_name}. Skipping.")
            continue
            
        # Determine how many to pick
        current_pick_count = num_to_pick_per_folder
        if current_pick_count > total_images:
            print(f"  -> Warning: Requested {current_pick_count}, but only {total_images} exist in {subfolder_name}.")
            print(f"  -> Copying all {total_images} images.")
            current_pick_count = total_images

        # Random Sampling
        selected_images = random.sample(images, current_pick_count)
        
        # Copy Process
        folder_success_count = 0
        for img_name in selected_images:
            src_file = os.path.join(current_src_path, img_name)
            dst_file = os.path.join(dst_folder, img_name)
            
            # Check if file already exists in destination to prevent overwrite (optional safety)
            if os.path.exists(dst_file):
                # Simple rename strategy if file exists: name_subfolder.jpg
                name, ext = os.path.splitext(img_name)
                new_name = f"{name}_{subfolder_name}{ext}"
                dst_file = os.path.join(dst_folder, new_name)

            try:
                shutil.copy2(src_file, dst_file)
                folder_success_count += 1
            except Exception as e:
                print(f"  -> Failed to copy {img_name}: {e}")
        
        grand_total_copied += folder_success_count
        print(f"  -> Copied {folder_success_count} images from {subfolder_name}")

    print("-" * 30)
    print(f"Process Complete.")
    print(f"Grand Total: {grand_total_copied} images copied to:\n{dst_folder}")

if __name__ == "__main__":
    copy_random_images_from_subfolders()