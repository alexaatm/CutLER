import datetime
import numpy as np
from PIL import Image
from pycocotools import mask as cocomask
import pycocotools.mask as cocomask_util
import argparse
import os
from tqdm import tqdm
import json

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    """
    Ref: from CutLER maskcut.py: https://github.com/facebookresearch/CutLER
    Return image_info in COCO style
    Args:
        image_id: the image ID
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        date_captured: the date this image info is created
        license: license of this image
        coco_url: url to COCO images if there is any
        flickr_url: url to flickr if there is any
    """
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }
    return image_info

def create_annotation_info(annotation_id, image_id, category_info, binary_mask, 
                           image_size=None, bounding_box=None):
    """
    Ref: from CutLER maskcut.py: https://github.com/facebookresearch/CutLER
    Return annotation info in COCO style
    Args:
        annotation_id: the annotation ID
        image_id: the image ID
        category_info: the information on categories
        binary_mask: a 2D binary numpy array where '1's represent the object
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        bounding_box: the bounding box for detection task. If bounding_box is not provided, 
        we will generate one according to the binary mask.
    """
    # TODO: check why need the threshold below - for image with values btw 0...1 it will binirize them
    # upper = np.max(binary_mask)
    # lower = np.min(binary_mask)
    # thresh = upper / 2.0
    # binary_mask[binary_mask > thresh] = upper
    # binary_mask[binary_mask <= thresh] = lower
    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask.astype(np.uint8), image_size)

    binary_mask_encoded = cocomask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    area = cocomask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = cocomask.toBbox(binary_mask_encoded)

    rle = cocomask_util.encode(np.array(binary_mask[...,None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    segmentation = rle

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": 0,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    } 

    return annotation_info


# necessay info used for coco style annotations
INFO = {
    "description": "Pseudo-masks",
    "url": "https://github.com/alexaatm/CutLER/tree/67c1d46494af70d306300db8ac195902c2fbce3d",
    "version": "1.0",
    "year": 2023,
    "contributor": "alexaatm",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Apache License",
        "url": "https://github.com/facebookresearch/CutLER/blob/main/LICENSE"
    }
]

# only one class, i.e. foreground
CATEGORIES = [
    {
        'id': 1,
        'name': 'fg',
        'supercategory': 'fg',
    },
]

output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []}

category_info = {
    "is_crowd": 0,
    "id": 1
}

def labelmap2masks(labelmap: np.array):
    unique_labels = np.unique(labelmap)
    if unique_labels.size == 0:
        # if no labels, cannot create a list of binary masks
        return None
    else:
        masks = []
        for label in unique_labels:
            mask = (labelmap==label).astype(np.uint8)
            masks.append(mask)
        return masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generate coco-style annotation from labelmaps')
    parser.add_argument('--labelmap_path', type=str,
                        default='/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/carotid-mini/labelmaps',
                        help='Path to labelmaps directory')
    parser.add_argument('--save_path', type=str, 
                        default='/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/carotid-mini/annotations/carotid-mini_train_fixsize480_tau0.15_N3.json',
                        help='Path to save generated coco-style annotation')
    
    args = parser.parse_args()

    # get list of labelmaps - it corresponds to the list of images
    labelmap_list = os.listdir(args.labelmap_path)

    # a list to track if we added an image name to images list
    image_names = []
    image_id, segmentation_id = 1, 1

    for labelmap_full in tqdm(labelmap_list):
        path = os.path.join(args.labelmap_path, labelmap_full)
        labelmap = np.array(Image.open(path).convert('L'))
        masks = labelmap2masks(labelmap)

        if masks==None:
            # no need to generate an annotation file
            print(f'Skipping empty labelmap: {labelmap_list}')
            continue

        # TODO: consider traversing dataset images folder, bc here it is assumed
        # that labelmaps have the same names and size as images
        # Upd: they don't, at least the extention... 
        name = os.path.splitext(os.path.basename(labelmap_full))[0]
        # To make it work with the dataset data/mutinfo_train_carotid/images/*.jpg
        # name = name + ".jpg"
        # To make it work with the dataset data/US_MIXED/train/images2/images/*.png (have folder of images inside images2 folder (that's what cutler expects))
        name = 'images/' + name + '.png'
        height, width = labelmap.shape

        # create coco-style image info
        if name not in image_names:
            image_info = create_image_info(image_id, name, (height, width, 3))
            output["images"].append(image_info)
            image_names.append(name)
        
        for mask in masks:
            # create coco-style annotation info created FOR EACH mask
            annotation_info = create_annotation_info(
                segmentation_id, image_id, category_info, mask.astype(np.uint8), None)
            if annotation_info is not None:
                output["annotations"].append(annotation_info)
                segmentation_id += 1
        image_id += 1

    json_name = args.save_path
    with open(json_name, 'w') as output_json_file:
        json.dump(output, output_json_file)
    print(f'dumping {json_name}')
    print("Done: {} images; {} anns.".format(len(output['images']), len(output['annotations'])))