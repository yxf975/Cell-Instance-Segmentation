import os
import cv2
import glob
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pycocotools.mask as mask_utils  # import encode, decode, frPyObjects, toBbox, area
from sklearn.model_selection import KFold
from itertools import product
from pycocotools.coco import COCO


def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, float) and np.isnan(enc):
            continue
        s = enc.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start:start + length] = 1 + m
    return img.reshape(shape)


def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


if __name__ == "__main__":
    file_names = glob.glob("../../data/train/*.png")
    labels = pd.read_csv("../../data/train.csv")

    all_categories = [{"id": (i + 1), "name": v} for i, v in enumerate(labels.cell_type.unique())]
    category_dict = {v: i + 1 for i, v in enumerate(labels.cell_type.unique())}
    print(category_dict)

    split = KFold(5, random_state=0, shuffle=True)
    img_ids = np.array(labels.id.unique())
    splits = list(split.split(img_ids))

    all_images = []
    all_annotations = []
    for fold in range(5):
        images = []
        annotations = []
        train_idx, valid_idx = splits[fold]
        train_img_ids = img_ids[train_idx]
        valid_img_ids = img_ids[valid_idx]
        print(labels[labels.id.isin(valid_img_ids)].cell_type.value_counts())
        for img_id in tqdm(valid_img_ids):
            group = labels[labels.id == img_id]
            height, width = (int(group.height.unique()[0]), int(group.width.unique()[0]))
            masks = group.annotation.tolist()
            cell_types = group.cell_type.tolist()
            assert len(set(cell_types)) == 1
            img = cv2.imread(os.path.join("../../data/train", img_id + ".png"))
            H, W = img.shape[:2]

            ms = []
            for mask, cell_type in zip(masks, cell_types):
                mask = np.asfortranarray(enc2mask([mask], (height, width)))

                ms.append(mask)
            ms = np.stack(ms, -1)
            ms_sum = ms.sum((0, 1))
            cuts = 4
            wstarts = W * np.arange(cuts).astype(int) // (cuts + 1)
            wends = W * np.arange(2, cuts + 2).astype(int) // (cuts + 1)
            hstarts = H * np.arange(cuts).astype(int) // (cuts + 1)
            hends = H * np.arange(2, cuts + 2).astype(int) // (cuts + 1)
            for i, j in product(range(cuts), range(cuts)):

                img_cut = img[hstarts[i]:hends[i], wstarts[j]:wends[j]]
                mask_cut = ms[hstarts[i]:hends[i], wstarts[j]:wends[j]]
                mask_cut = mask_cut[..., mask_cut.sum((0, 1)) > 0.25 * ms_sum]
                cv2.imwrite(os.path.join("../../data/train_tiny/images", f"{img_id}_{i}_{j}.jpg"), img_cut)
                images.append({
                    "id": f"{img_id}_{i}_{j}",
                    "file_name": f"{img_id}_{i}_{j}.jpg",
                    "width": img_cut.shape[1],
                    "height": img_cut.shape[0]
                })
                for l in range(mask_cut.shape[-1]):
                    rle = mask_utils.encode(mask_cut[..., l])
                    rle['counts'] = rle['counts'].decode()
                    bbox = [int(_) for _ in mask_utils.toBbox(rle)]
                    area = mask_utils.area(rle)
                    annotations.append({
                        "id": len(annotations),
                        "image_id": f"{img_id}_{i}_{j}",
                        "category_id": category_dict[cell_types[0]],
                        "bbox": bbox,
                        "segmentation": rle,
                        "iscrowd": 0,
                        "area": int(area)
                    })

        with open(f"../../data/train_tiny/annotations/fold_{fold}.json", "w") as f:
            json.dump({"images": images, "annotations": annotations, "categories": all_categories}, f)
    #         json.dump({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "cort"}]}, f)

    # annFile = '../data/train_tiny/annotations/fold_0.json'
    # coco = COCO(annFile)
    # imgIds = coco.getImgIds()
    #
    # _, axs = plt.subplots(2, 2, figsize=(8, 6))
    # for imgid, ax in zip(imgIds[:2], axs):
    #     img = cv2.imread("../data/train_tiny" + f'/images/{imgid}.jpg')
    #     #     img_img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(img[...,0])
    #     annIds = coco.getAnnIds(imgIds=[imgid])
    #     anns = coco.loadAnns(annIds)
    #     ax[0].imshow(img)
    #     ax[1].imshow(img)
    #     plt.sca(ax[1])
    #     coco.showAnns(anns, draw_bbox=True)
    #
    # plt.tight_layout()
    # plt.show()
