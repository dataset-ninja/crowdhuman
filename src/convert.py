# https://www.crowdhuman.org/

import glob
import json
import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from dotenv import load_dotenv
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s
from dataset_tools.convert import unpack_if_archive


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "crowdhuman"
    batch_size = 30
    images_ext = ".jpg"

    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        im_name = get_file_name(image_path)

        bboxes_data = id_to_bboxes[im_name]
        for curr_bbox_data in bboxes_data:
            fbox_check = None
            for curr_box in ["fbox", "vbox", "hbox"]:
                box = curr_bbox_data.get(curr_box)
                if curr_box == "fbox":
                    fbox_check = box
                else:
                    if box == fbox_check:
                        continue
                if box is not None:
                    top = box[1]
                    left = box[0]
                    bottom = box[1] + box[3]
                    right = box[0] + box[2]

                    if top >= bottom or left >= right:
                        print(box)
                        continue

                    rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)

                    label_rectangle = sly.Label(rectangle, name_to_class[curr_box])
                    labels.append(label_rectangle)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)

    obj_class_head = sly.ObjClass("head", sly.Rectangle)
    obj_class_visible = sly.ObjClass("visible", sly.Rectangle)
    obj_class_full = sly.ObjClass("full-body", sly.Rectangle)
    name_to_class = {"fbox": obj_class_full, "vbox": obj_class_visible, "hbox": obj_class_head}

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[obj_class_head, obj_class_visible, obj_class_full])
    api.project.update_meta(project.id, meta.to_json())

    for ds_name in ["train", "val", "test"]:
        id_to_bboxes = defaultdict(list)
        if ds_name == "val":
            images_path = "APP_DATA/CrowdHuman/CrowdHuman_val/Images"
            images_names = os.listdir(images_path)
            images_pathes = [os.path.join(images_path, image_name) for image_name in images_names]
            bboxes_file_path = "APP_DATA/CrowdHuman/annotation_val.odgt"

        elif ds_name == "train":
            images_pathes = glob.glob("APP_DATA/CrowdHuman" + "/*train0*/*/*.jpg")
            bboxes_file_path = "APP_DATA/CrowdHuman/annotation_train.odgt"

        elif ds_name == "test":
            images_path = "APP_DATA/CrowdHuman/CrowdHuman_test/images_test"
            images_names = os.listdir(images_path)
            images_pathes = [os.path.join(images_path, image_name) for image_name in images_names]

        if ds_name != "test":
            with open(bboxes_file_path) as f:
                content = f.read().split("\n")
                for row in content:
                    if len(row) != 0:
                        curr_data = json.loads(row)
                        id_to_bboxes[curr_data["ID"]] = curr_data["gtboxes"]

        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_pathes))

        for img_pathes_batch in sly.batched(images_pathes, batch_size=batch_size):
            img_names_batch = [
                get_file_name_with_ext(image_path) for image_path in img_pathes_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            if ds_name != "test":
                anns_batch = [create_ann(image_path) for image_path in img_pathes_batch]
                api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))

    return project
