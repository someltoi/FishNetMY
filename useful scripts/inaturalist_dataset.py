from pyinaturalist import get_observations
import requests
import os
from PIL import Image
from io import BytesIO
import csv

# -------------------------------
# USER CONFIGURATIONS
# -------------------------------
species_list = [
                "Epinephelus coioides", 
                "Epinephelus tauvina", 
                "Epinephelus fuscoguttatus",
                "Lates calcarifer", 
                "Oreochromis niloticus",
                "Oreochromis mossambicus",
                "Oreochromis aureus"
                ]

# Acceptable image licenses
allowed_licenses = {

    # Add cc-by-nc **ONLY** if your work is non-commercial
    "cc-by-nc",
}

# Minimum required dimensions (width, height)
min_width =300
min_height = 300

output_dir = "/home/somel/code/FYP_Project/Dataset/iNaturalist/version2/"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# FETCH OBSERVATIONS
# -------------------------------
count = 0

def download_dataset(data,output_dir,min_width,min_height,allowed_licenses,species,count):
    species_dir = os.path.join(output_dir, species.replace(" ", "_"))
    os.makedirs(species_dir, exist_ok=True)
    
    for result in data["results"]:
        obs_id = result["id"]
        obs_license = result.get("license_code", "").lower()

        # Filter by observation license
        if obs_license not in allowed_licenses:
            continue
        
        for photo in result.get("photos", []):
            photo_license = photo.get("license_code", "").lower()

            # Filter by photo license
            if photo_license not in allowed_licenses:
                continue

            # Get original-size image
            url = photo["url"].replace("square", "original")

            # Download image
            try:
                response = requests.get(url, timeout=10)
                img = Image.open(BytesIO(response.content))
            except Exception:
                continue

            # Check resolution
            w, h = img.size
            if w < min_width or h < min_height:
                continue

            # Save image
            filename = os.path.join(species_dir, f"{species}_{count}.jpg")
            
            # Convert RGBA or P-mode images to RGB before saving
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            img.save(filename)

            # Save metadata
            print(f"Saved: {filename} (Obs ID: {obs_id}, License: {photo_license}, Size: {w}x{h})")
            count += 1

    print(f"Download complete: {count} images meeting license and size requirements")

for species in species_list:
    data = get_observations(
        taxon_name=species,
        per_page=400,
        licensed=True,
        photo_licensed=True
    )
    download_dataset(data,output_dir,min_width,min_height,allowed_licenses,species,count)
