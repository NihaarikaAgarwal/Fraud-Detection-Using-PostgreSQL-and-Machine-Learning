import os
import base64
import csv
from pathlib import Path

# --------------------------- CONFIGURATION --------------------------- #
FOLDER_PATH = "/Users/nihaarikaagarwal/Downloads/EST/fraud6_crop_and_replace"   
OUTPUT_CSV  = "/Users/nihaarikaagarwal/Downloads/EST/fraud_crop_replace_encoded_images.csv"    
NUM_IMAGES  = 3000                     
ALLOWED_EXT = {".png", ".jpg", ".jpeg"}

# --------------------------------------------------------------------- #

def encode_images_to_csv(folder_path, output_csv, num_images=None):
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not valid")

    image_paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in ALLOWED_EXT])

    if num_images is not None:
        image_paths = image_paths[:num_images]

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ImageName", "Base64", "TRUE", "Type"])

        for img_path in image_paths:
            with open(img_path, "rb") as img_file:
                b64_str = base64.b64encode(img_file.read()).decode("utf-8")

            writer.writerow([img_path.name, b64_str, "TRUE", "crop_and_replace"])

    print(f" Wrote {len(image_paths)} rows to {output_csv}")

if __name__ == "__main__":
    encode_images_to_csv(FOLDER_PATH, OUTPUT_CSV, NUM_IMAGES)
