import os

def rename_images():
    print("--- Batch Image Renamer ---")

    # --- CONFIGURATION (Edit these) ---
    # 1. The folder containing the images
    target_folder = "/workspaces/ai-projects/Dataset/Malaysian maritime fish/version 2/Seabass"
    
    # 2. The new name to use (e.g. 'tilapia' becomes 'tilapia_1.jpg')
    base_name = "seabass"
    
    # 3. Starting number for the count
    start_count = 1
    
    # ----------------------------------

    if not os.path.exists(target_folder):
        print(f"Error: Folder '{target_folder}' does not exist.")
        return

    # Extensions to look for
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    # Get list of files and sort them (so the order is consistent)
    files = os.listdir(target_folder)
    images = [f for f in files if f.lower().endswith(valid_extensions)]
    images.sort()

    if not images:
        print("No images found to rename.")
        return

    print(f"Found {len(images)} images in '{os.path.basename(target_folder)}'. Renaming...")

    renamed_count = 0
    current_count = start_count

    for filename in images:
        # Get original extension (e.g., .jpg)
        name, ext = os.path.splitext(filename)
        
        # Define the new filename
        new_filename = f"{base_name}_{current_count}{ext}"
        
        # Full paths
        old_path = os.path.join(target_folder, filename)
        new_path = os.path.join(target_folder, new_filename)

        # Check if file is already named correctly
        if filename == new_filename:
            current_count += 1
            continue
            
        # Check if the destination filename already exists (prevent overwrite)
        if os.path.exists(new_path):
            print(f"⚠️ Skipping '{filename}': '{new_filename}' already exists.")
            # You can choose to increment here if you want to skip that number
            # current_count += 1 
            continue

        try:
            os.rename(old_path, new_path)
            # Optional: print every rename
            # print(f"{filename} -> {new_filename}")
            renamed_count += 1
            current_count += 1
        except OSError as e:
            print(f"Error renaming {filename}: {e}")

    print("-" * 30)
    print(f"Success! Renamed {renamed_count} images.")

if __name__ == "__main__":
    rename_images()