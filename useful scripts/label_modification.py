import os

# --- CONFIGURATION ---
LABEL_FOLDER_PATH = r"/workspaces/ai-projects/Dataset/Nile Tilapia.v1i.yolov11/train/labels" # Put your path here
OLD_CLASS_ID = 0  # The ID for Tilapia in the 2nd dataset
NEW_CLASS_ID = 2  # The ID for Tilapia in the 1st dataset (Target)

def update_classes():
    # Loop through all files in the directory
    for filename in os.listdir(LABEL_FOLDER_PATH):
        if filename.endswith(".txt"):
            filepath = os.path.join(LABEL_FOLDER_PATH, filename)
            
            new_lines = []
            file_changed = False
            
            # Read the file
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            # Process each object in the image
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = int(parts[0])
                    
                    # Check if this line needs changing
                    if class_id == OLD_CLASS_ID:
                        parts[0] = str(NEW_CLASS_ID)
                        file_changed = True
                    
                    # Reconstruct the line
                    new_lines.append(" ".join(parts) + "\n")
            
            # Overwrite the file only if changes were made
            if file_changed:
                with open(filepath, 'w') as f:
                    f.writelines(new_lines)
                print(f"Updated: {filename}")

if __name__ == "__main__":
    update_classes()
    print("Conversion complete.")