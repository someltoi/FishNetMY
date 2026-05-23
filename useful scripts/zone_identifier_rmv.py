import os

# --- CONFIGURATION ---
TARGET_FOLDER = "/home/somel/code/FYP_Project/Dataset/YOLO"  # Change this path

def remove_zone_identifiers():
    deleted_count = 0
    
    # Walk through all folders and subfolders
    for root, dirs, files in os.walk(TARGET_FOLDER):
        for filename in files:
            # Check if the file is a Zone.Identifier file
            if "Zone.Identifier" in filename:
                file_path = os.path.join(root, filename)
                
                try:
                    os.remove(file_path)
                    print(f"Deleted: {filename}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")

    print(f"--- Process Complete. Removed {deleted_count} files. ---")

if __name__ == "__main__":
    remove_zone_identifiers()