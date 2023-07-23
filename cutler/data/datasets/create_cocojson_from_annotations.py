import os
import json
import cv2
from pycocotools.coco import COCO as coco
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pycocotools import mask as maskUtils

def convert_to_coco_format(image_folder, annotation_folder, output_file):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Define categories (adjust as needed)
    categories = [
        {"id": 0, "name": "category_0"}, #background
        {"id": 1, "name": "category_1"}, #carotid artery
        {"id": 2, "name": "category_2"},
        {"id": 3, "name": "category_3"},
        {"id": 4, "name": "category_4"},
        {"id": 5, "name": "category_5"},
        # Add more categories as needed
    ]
    coco_data["categories"] = categories

    image_id = 0
    annotation_id = 0

    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):  # Adjust the file extension according to your images
            image_id += 1
            image_filename = os.path.join(image_folder, filename)
            annotation_filename = os.path.join(annotation_folder, filename.replace(".jpg", ".png"))

            # Load the image and annotation
            image = cv2.imread(image_filename)
            annotation = cv2.imread(annotation_filename, cv2.IMREAD_GRAYSCALE)

            # Get the height and width of the image
            height, width = image.shape[:2]

            # Create the "images" entry in the COCO format
            coco_image = {
                "id": image_id,
                "file_name": filename,
                "height": height,
                "width": width
            }
            coco_data["images"].append(coco_image)

            # Create the segmentation mask for the entire image
            segmentation_mask = annotation.astype("uint8")

            # Make the mask Fortran contiguous
            segmentation_mask = np.asfortranarray(segmentation_mask)

            # Encode the segmentation mask using RLE (Run-Length Encoding)
            rle_encoded = maskUtils.encode(segmentation_mask)
            rle_encoded["counts"] = rle_encoded["counts"].decode("utf-8")  # Convert bytes to string

            # Create the "annotations" entry in the COCO format
            coco_annotation = {
                "id": image_id,
                "image_id": image_id,
                "category_id": 0,  # Assuming you have only one category (index 0)
                "segmentation": rle_encoded,
                "area": float(cv2.countNonZero(segmentation_mask)),
                "bbox": [],  # No bounding box for semantic segmentation
                "iscrowd": 0
            }
            coco_data["annotations"].append(coco_annotation)

    # Save the COCO format data to a JSON file
    with open(output_file, "w") as f:
        json.dump(coco_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert annotations to COCO format")
    parser.add_argument("--image_folder", type=str, help="Path to the folder containing the images")
    parser.add_argument("--annotation_folder", type=str, help="Path to the folder containing the annotations")
    parser.add_argument("--output_file", type=str, help="Path to the output COCO annotations JSON file")
    args = parser.parse_args()
    convert_to_coco_format(args.image_folder, args.annotation_folder, args.output_file)
