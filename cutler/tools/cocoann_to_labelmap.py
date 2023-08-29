import os
import json
import tqdm
import datetime
import argparse
import pycocotools.mask as cocomask
from detectron2.utils.file_io import PathManager
import numpy as np
from PIL import Image
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def segmToRLE(segm, h, w):
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = cocomask.frPyObjects(segm, h, w)
        rle = cocomask.merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = cocomask.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle

def rle2mask(rle, height, width):
    if "counts" in rle and isinstance(rle["counts"], list):
        # if compact RLE, ignore this conversion
        # Magic RLE format handling painfully discovered by looking at the
        # COCO API showAnns function.
        rle = cocomask.frPyObjects(rle, height, width)
    mask = cocomask.decode(rle)
    return mask

def cocosegm2mask(segm, h, w):
    rle = segmToRLE(segm, h, w)
    mask = rle2mask(rle, h, w)
    return mask

def masks2labelmap(masks: list, h: int, w: int):
    labelmap = np.zeros((h, w), dtype=int) #Check if order h, w is correct
    label = 1 #start from 1, to leave 0 be a background index in the labelmap
    for mask in masks:
        labelmap[mask==1] = label
        label = label + 1
    return labelmap


if __name__ == "__main__":
# load model arguments
    parser = argparse.ArgumentParser(description='Generate labelmaps from json files')
    parser.add_argument('--ann_path', type=str, 
                        default='/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/carotid-mini/annotations/imagenet_train_fixsize480_tau0.15_N3.json',
                        help='Path to maskcut annotation or model predictions')
    parser.add_argument('--save_path', type=str,
                        default='/home/guests/oleksandra_tmenova/test/project/thesis-codebase/data/carotid-mini/labelmaps',
                        help='Path to save the generated labelmaps')
    # parser.add_argument('--threshold', type=float, default=0.5,
    #                     help='Confidence score thresholds')
    args = parser.parse_args()


    # check if directory for saving files exists
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)  

    # load annotations
    ann_dict = json.load(open(args.ann_path))
    # with PathManager.open(ann_path, "r") as f:
        # ann_dict = json.load(f)
    image_list = ann_dict['images']
    annotations = ann_dict['annotations']

    # create id to filename mapping (to correctly name labelmaps)
    id_to_filename = {}
    for image_data in image_list:
        full_file_name = image_data['file_name']
        base_name = os.path.splitext(os.path.basename(full_file_name))[0]
        id_to_filename[image_data['id']] = base_name

    # group annotations by images in a new dictionary
    image_to_anns = {}
    for id, ann in enumerate(annotations):
        if ann['image_id'] in image_to_anns:
            image_to_anns[ann['image_id']].append(ann)
        else:
            image_to_anns[ann['image_id']] = [ann]

    # dicrionary for storing labelmaps
    image_to_labelmaps = {}

    for k, anns in tqdm.tqdm(image_to_anns.items()):
        masks = []
        for ann in anns:
            segm = ann['segmentation']
            mask = cocosegm2mask(segm, segm['size'][0], segm['size'][1])
            masks.append(mask)

        # since anns are grouped per image, can get h,w, name from any ann in the list
        h = anns[0]['height']
        w = anns[0]['width']
        # TODO: figure out a better way to get the actual filename of th eimage (stored in image_list)
        # E.g. can build the original im_ann dictionary using image names, not ids
        image_id = anns[0]['image_id']
        image_name = id_to_filename[image_id]
        

        # generate labelmaps
        labelmap = masks2labelmap(masks, h, w)
        image_to_labelmaps[k] = labelmap

        # save labelmap
        output_file = f'{args.save_path}/{image_name}.png'
        labelmap_im = Image.fromarray(labelmap.astype(np.uint8)).convert('L')
        labelmap_im.save(output_file)
