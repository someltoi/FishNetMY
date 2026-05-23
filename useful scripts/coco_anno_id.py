import json


train_annotations_file = '/home/somel/code/FYP_Project/Dataset/COCO/version_4/train/_annotations.coco.json'
val_annotations_file = '/home/somel/code/FYP_Project/Dataset/COCO/version_4/valid/_annotations.coco.json'
test_annotations_file = '/home/somel/code/FYP_Project/Dataset/COCO/version_4/test/_annotations.coco.json'

def fix_coco_ids(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if fix is needed
    if any(cat['id'] == 0 for cat in data['categories']):
        print(f"Fixing Class 0 issue in {json_path}...")
        
        # Shift Categories
        for cat in data['categories']:
            cat['id'] += 1
            
        # Shift Annotations
        for ann in data['annotations']:
            ann['category_id'] += 1
            
        with open(json_path, 'w') as f:
            json.dump(data, f)
        print("Fixed.")
    else:
        print(f"{json_path} already correct.")

# Run this on all your files
fix_coco_ids(train_annotations_file)
fix_coco_ids(val_annotations_file)
fix_coco_ids(test_annotations_file)