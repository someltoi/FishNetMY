import json
import os

def merge_coco_datasets(dataset_a_path, dataset_b_path, output_path):
    print(f"Loading {dataset_a_path}...")
    with open(dataset_a_path, 'r') as f:
        data_a = json.load(f)

    print(f"Loading {dataset_b_path}...")
    with open(dataset_b_path, 'r') as f:
        data_b = json.load(f)

    merged_data = {
        "info": data_a.get("info", {}),
        "licenses": data_a.get("licenses", []),
        "categories": [],
        "images": [],
        "annotations": []
    }

    # ---------------------------------------------------
    # 1. MERGE CATEGORIES — EXCLUDE ID=0 PLACEHOLDER CATEGORIES
    # ---------------------------------------------------

    VALID_CLASSES = {"Tilapia", "Seabass", "Groupers"}

    def filter_real_categories(categories):
        """Remove any category where id==0 or name is an artificial parent."""
        filtered = []
        for c in categories:
            if c["id"] == 0:
                continue
            if c["name"] not in VALID_CLASSES:
                continue
            filtered.append(c)
        return filtered

    cats_a = filter_real_categories(data_a["categories"])
    cats_b = filter_real_categories(data_b["categories"])

    category_map_a = {}
    category_map_b = {}
    unified_categories = {}
    next_cat_id = 0

    print("Merging categories...")

    # Process categories from dataset A
    for cat in cats_a:
        name = cat["name"]
        if name not in unified_categories:
            unified_categories[name] = next_cat_id
            merged_data["categories"].append({
                "id": next_cat_id,
                "name": name,
                "supercategory": "fish"
            })
            next_cat_id += 1

        category_map_a[cat["id"]] = unified_categories[name]

    # Process categories from dataset B
    for cat in cats_b:
        name = cat["name"]
        if name not in unified_categories:
            unified_categories[name] = next_cat_id
            merged_data["categories"].append({
                "id": next_cat_id,
                "name": name,
                "supercategory": "fish"
            })
            next_cat_id += 1

        category_map_b[cat["id"]] = unified_categories[name]

    print(f"Unified Categories: {unified_categories}")

    # ---------------------------------------------------
    # 2. MERGE IMAGES
    # ---------------------------------------------------
    print("Merging images...")

    merged_data["images"].extend(data_a["images"])
    max_img_id = max([img["id"] for img in data_a["images"]]) if data_a["images"] else 0

    image_id_map_b = {}

    for img in data_b["images"]:
        old_id = img["id"]
        max_img_id += 1
        new_id = max_img_id

        image_id_map_b[old_id] = new_id

        new_img = img.copy()
        new_img["id"] = new_id
        merged_data["images"].append(new_img)

    # ---------------------------------------------------
    # 3. MERGE ANNOTATIONS — EXCLUDE ANNOTATIONS OF REMOVED CATEGORIES
    # ---------------------------------------------------
    print("Merging annotations...")

    # Dataset A
    for ann in data_a["annotations"]:
        old_cat = ann["category_id"]
        if old_cat not in category_map_a:
            continue  # skip invalid categories

        new_ann = ann.copy()
        new_ann["category_id"] = category_map_a[old_cat]
        merged_data["annotations"].append(new_ann)

    max_ann_id = max([ann["id"] for ann in data_a["annotations"]]) if data_a["annotations"] else 0

    # Dataset B
    for ann in data_b["annotations"]:
        old_cat = ann["category_id"]
        old_img = ann["image_id"]

        if old_cat not in category_map_b:
            continue
        if old_img not in image_id_map_b:
            continue

        max_ann_id += 1

        new_ann = ann.copy()
        new_ann["id"] = max_ann_id
        new_ann["image_id"] = image_id_map_b[old_img]
        new_ann["category_id"] = category_map_b[old_cat]

        merged_data["annotations"].append(new_ann)

    # ---------------------------------------------------
    # 4. SAVE OUTPUT
    # ---------------------------------------------------
    print(f"Saving merged dataset to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print("Done! Summary:")
    print(f"  Images: {len(merged_data['images'])}")
    print(f"  Annotations: {len(merged_data['annotations'])}")
    print(f"  Categories: {len(merged_data['categories'])}")


# --- USAGE ---
file_a = '/home/somel/code/FYP_Project/Dataset/COCO/Malaysia Fish Object Detection.v5-hopefully-more-balanced.coco/valid/_annotations.coco.json'
file_b = '/home/somel/code/FYP_Project/Dataset/COCO/version2.v6-without-tilapia.coco/valid/_annotations.coco.json'
output_file = '_annotations.coco.json'

merge_coco_datasets(file_a, file_b, output_file)
