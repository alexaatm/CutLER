# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/builtin.py

"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from .builtin_meta import _get_builtin_metadata
from .coco import register_coco_instances

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO_SEMI = {}
_PREDEFINED_SPLITS_COCO_SEMI["coco_semi"] = {
    # we use seed 42 to be consistent with previous works on SSL detection and segmentation
    "coco_semi_1perc": ("coco/train2017", "coco/annotations/1perc_instances_train2017.json"),
    "coco_semi_2perc": ("coco/train2017", "coco/annotations/2perc_instances_train2017.json"),
    "coco_semi_5perc": ("coco/train2017", "coco/annotations/5perc_instances_train2017.json"),
    "coco_semi_10perc": ("coco/train2017", "coco/annotations/10perc_instances_train2017.json"),
    "coco_semi_20perc": ("coco/train2017", "coco/annotations/20perc_instances_train2017.json"),
    "coco_semi_30perc": ("coco/train2017", "coco/annotations/30perc_instances_train2017.json"),
    "coco_semi_40perc": ("coco/train2017", "coco/annotations/40perc_instances_train2017.json"),
    "coco_semi_50perc": ("coco/train2017", "coco/annotations/50perc_instances_train2017.json"),
    "coco_semi_60perc": ("coco/train2017", "coco/annotations/60perc_instances_train2017.json"),
    "coco_semi_80perc": ("coco/train2017", "coco/annotations/80perc_instances_train2017.json"),
}

_PREDEFINED_SPLITS_COCO_CA = {}
_PREDEFINED_SPLITS_COCO_CA["coco_cls_agnostic"] = {
    "cls_agnostic_coco": ("coco/val2017", "coco/annotations/coco_cls_agnostic_instances_val2017.json"),
    "cls_agnostic_coco20k": ("coco/train2014", "coco/annotations/coco20k_trainval_gt.json"),
}

_PREDEFINED_SPLITS_IMAGENET = {}
_PREDEFINED_SPLITS_IMAGENET["imagenet"] = {
    # maskcut annotations
    "imagenet_train": ("imagenet/train", "imagenet/annotations/imagenet_train_fixsize480_tau0.15_N3.json"),
    # self-training round 1
    "imagenet_train_r1": ("imagenet/train", "imagenet/annotations/cutler_imagenet1k_train_r1.json"),
    # self-training round 2
    "imagenet_train_r2": ("imagenet/train", "imagenet/annotations/cutler_imagenet1k_train_r2.json"),
    # self-training round 3
    "imagenet_train_r3": ("imagenet/train", "imagenet/annotations/cutler_imagenet1k_train_r3.json"),
}

_PREDEFINED_SPLITS_VOC = {}
_PREDEFINED_SPLITS_VOC["voc"] = {
    'cls_agnostic_voc': ("voc/", "voc/annotations/trainvaltest_2007_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_CROSSDOMAIN = {}
_PREDEFINED_SPLITS_CROSSDOMAIN["cross_domain"] = {
    'cls_agnostic_clipart': ("clipart/", "clipart/annotations/traintest_cls_agnostic.json"),
    'cls_agnostic_watercolor': ("watercolor/", "watercolor/annotations/traintest_cls_agnostic.json"),
    'cls_agnostic_comic': ("comic/", "comic/annotations/traintest_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_KITTI = {}
_PREDEFINED_SPLITS_KITTI["kitti"] = {
    'cls_agnostic_kitti': ("kitti/", "kitti/annotations/trainval_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_LVIS = {}
_PREDEFINED_SPLITS_LVIS["lvis"] = {
    "cls_agnostic_lvis": ("coco/", "coco/annotations/lvis1.0_cocofied_val_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_OBJECTS365 = {}
_PREDEFINED_SPLITS_OBJECTS365["objects365"] = {
    'cls_agnostic_objects365': ("objects365/val", "objects365/annotations/zhiyuan_objv2_val_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_OpenImages = {}
_PREDEFINED_SPLITS_OpenImages["openimages"] = {
    'cls_agnostic_openimages': ("openImages/validation", "openImages/annotations/openimages_val_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_UVO = {}
_PREDEFINED_SPLITS_UVO["uvo"] = {
    "cls_agnostic_uvo": ("uvo/all_UVO_frames", "uvo/annotations/val_sparse_cleaned_cls_agnostic.json"),
}

_PREDEFINED_SPLITS_carotidmini = {}
_PREDEFINED_SPLITS_carotidmini["carotid-mini"] = {
    'carotid-mini_train': ("carotid-mini/images", "carotid-mini/annotations/imagenet_train_fixsize480_tau0.15_N3.json"),
}

_PREDEFINED_SPLITS_fullcarotid = {}
_PREDEFINED_SPLITS_fullcarotid["full_carotid"] = {
    # maskcut annotations
    'full_carotid_train': ("full_carotid/images/train", "full_carotid/annotations/merged_imagenet_train_fixsize480_tau0.15_N3.json"),
}

_PREDEFINED_SPLITS_mutinfovalcarotid = {}
_PREDEFINED_SPLITS_mutinfovalcarotid["mutinfo_val_carotid"] = {
    # maskcut annotations
    'mutinfo_val_carotid_main': ("mutinfo_val_carotid/images2", "mutinfo_val_carotid/annotations/imagenet_train_fixsize480_tau0.15_N3.json"),
}

_PREDEFINED_SPLITS_mutinfotraincarotid = {}
_PREDEFINED_SPLITS_mutinfotraincarotid["mutinfo_train_carotid"] = {
    # maskcut annotations
    'mutinfo_train_carotid_main': ("mutinfo_train_carotid/images2", "mutinfo_train_carotid/annotations/maskcut_fixsize480_tau0.15_N3.json"),
    # self-training round 1
    "mutinfo_train_carotid_train_r1": ("mutinfo_train_carotid/images2", "mutinfo_train_carotid/annotations/cutler_mutinfo_train_r1.json"),
    # self-training round 2
    # self-training round 3
    # deep spectral annotation
    'mutinfo_train_carotid_dsp_main': ("mutinfo_train_carotid/images", "mutinfo_train_carotid/annotations/dsp_labelmaps_clusters15_dino_ssd0_crf_segmaps_jpgformat.json"),
    'mutinfo_train_carotid_dsp_dino_ssd1_crf_segmaps': ("mutinfo_train_carotid/images", "mutinfo_train_carotid/annotations/dsp_labelmaps_clusters15_dino_ssd1_crf_segmaps_jpgformat.json"),
    'mutinfo_train_carotid_dsp_dino_ssd2_crf_multi_region': ("mutinfo_train_carotid/images", "mutinfo_train_carotid/annotations/dsp_labelmaps_clusters15_dino_ssd2_crf_multi_region_jpgformat.json"),

}

_PREDEFINED_SPLITS_usmixed = {}
_PREDEFINED_SPLITS_usmixed["us_mixed"] = {
    # maskcut annotations
    'us_mixed_train1': ("US_MIXED/train/images2", "US_MIXED/train/annotations/imagenet_train_fixsize480_tau0.15_N3.json"),
    'us_mixed_test1': ("US_MIXED/test/images2", "US_MIXED/test/annotations/imagenet_train_fixsize480_tau0.15_N3.json"),
    'us_mixed_val1': ("US_MIXED/val/images2", "US_MIXED/val/annotations/imagenet_train_fixsize480_tau0.15_N3.json"),
    # for self training after unsupervised training
    'us_mixed_train1_r1': ("US_MIXED/train/images2", "US_MIXED/train/annotations/cutler_maskcut_cutler_us_mixed_model_final_thresh0.0_r1.json"),    
    # dsp annotation - TODO (train, val, test), and separate ones as well
    'us_mixed_train_dsp': ("US_MIXED/train/images2", "US_MIXED/train/annotations/dsp_masks_clusters15_dino1.0_ssd1.0_var0.0_crf_segmaps_fixed.json"),
    'us_mixed_val_dsp': ("US_MIXED/val/images2", "US_MIXED/val/annotations/dsp_masks_clusters15_dino1.0_ssd1.0_var0.0_crf_segmaps.json"),

}


def register_all_imagenet(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_IMAGENET.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_voc(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_VOC.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_cross_domain(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_CROSSDOMAIN.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_kitti(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_KITTI.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_objects365(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_OBJECTS365.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_openimages(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_OpenImages.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_uvo(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_UVO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_coco_semi(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO_SEMI.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_coco_ca(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO_CA.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_carotidmini(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_carotidmini.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_fullcarotid(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_fullcarotid.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_mutinfovalcarotid(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_mutinfovalcarotid.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_mutinfotraincarotid(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_mutinfotraincarotid.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_all_usmixed(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_usmixed.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
register_all_coco_semi(_root)
register_all_coco_ca(_root)
register_all_imagenet(_root)
register_all_uvo(_root)
register_all_voc(_root)
register_all_cross_domain(_root)
register_all_kitti(_root)
register_all_openimages(_root)
register_all_objects365(_root)
register_all_lvis(_root)
register_all_carotidmini(_root)
register_all_fullcarotid(_root)
register_all_mutinfovalcarotid(_root)
register_all_mutinfotraincarotid(_root)
register_all_usmixed(_root)

