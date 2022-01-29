import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from pathlib import Path
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as maskUtils
from joblib import Parallel, delayed
import json, itertools
import shutil

FOLD_VAL = 0
FOLD_TEST = [1, 2]
dataDir = Path('../data/train')
targetDataDir = "../data/sartorius_coco_dataset"


# Based on: https://www.kaggle.com/eigrad/convert-rle-to-bounding-box-x0-y0-x1-y1
def rle2mask(rle, img_w, img_h):
    # transforming the string into an array of shape (2, N)
    array = np.fromiter(rle.split(), dtype=np.uint)
    array = array.reshape((-1, 2)).T
    array[0] = array[0] - 1

    # decompressing the rle encoding (ie, turning [3, 1, 10, 2] into [3, 4, 10, 11, 12])
    # for faster mask construction
    starts, lenghts = array
    mask_decompressed = np.concatenate([np.arange(s, s + l, dtype=np.uint) for s, l in zip(starts, lenghts)])

    # Building the binary mask
    msk_img = np.zeros(img_w * img_h, dtype=np.uint8)
    msk_img[mask_decompressed] = 1
    msk_img = msk_img.reshape((img_h, img_w))
    msk_img = np.asfortranarray(msk_img)  # This is important so pycocotools can handle this object

    return msk_img


def annotate(idx, row, cat_ids):
    mask = rle2mask(row['annotation'], row['width'], row['height'])  # Binary mask
    c_rle = maskUtils.encode(mask)  # Encoding it back to rle (coco format)
    c_rle['counts'] = c_rle['counts'].decode('utf-8')  # converting from binary to utf-8
    area = maskUtils.area(c_rle).item()  # calculating the area
    bbox = maskUtils.toBbox(c_rle).astype(int).tolist()  # calculating the bboxes
    annotation = {
        'segmentation': c_rle,
        'bbox': bbox,
        'area': area,
        'image_id': row['id'],
        'category_id': cat_ids[row['cell_type']],  # cat_ids[row['cell_type']],
        'iscrowd': 0,
        'id': idx
    }
    return annotation


def coco_structure(df, workers=4):
    # Building the header
    cat_ids = {'shsy5y': 1, "astro": 2, "cort": 3}
    cats = [{'name': name, 'id': id} for name, id in cat_ids.items()]
    images = [{'id': id, 'width': row.width, 'height': row.height, 'file_name': f'{id}.png'} \
              for id, row in df.groupby('id').agg('first').iterrows()]

    # Building the annotations
    annotations = Parallel(n_jobs=workers)(
        delayed(annotate)(idx, row, cat_ids) for idx, row in tqdm(df.iterrows(), total=len(df)))

    return {'categories': cats, 'images': images, 'annotations': annotations}


def run_copy(row):
    img_path = dataDir / f'{row.id}.png'
    if row.fold == FOLD_VAL:
        shutil.copy(img_path, targetDataDir + '/valid/')
    elif row.fold in FOLD_TEST:
        shutil.copy(img_path, targetDataDir + '/test/')
    else:
        shutil.copy(img_path, targetDataDir + '/train/')


if __name__ == "__main__":
    df = pd.read_csv('../data/train.csv')
    print(df.head())

    df = df.reset_index(drop=True)
    df['fold'] = -1
    skf = GroupKFold(n_splits=10)
    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df['cell_type'], groups=df['id'])):
        df.loc[val_idx, 'fold'] = fold

    train_df = df.query("fold!=@FOLD_VAL and fold not in @FOLD_TEST ")
    valid_df = df.query("fold==@FOLD_VAL")
    test_df = df.query("fold in @FOLD_TEST")
    print("length:", len(train_df), len(valid_df), len(test_df))

    train_json = coco_structure(train_df)
    valid_json = coco_structure(valid_df)
    test_json = coco_structure(test_df)

    print(train_json['annotations'][0])
    print(valid_json['annotations'][0])
    print(test_json['annotations'][0])

    with open(targetDataDir + '/annotations_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_json, f, ensure_ascii=True, indent=4)
    with open(targetDataDir + '/annotations_valid.json', 'w', encoding='utf-8') as f:
        json.dump(valid_json, f, ensure_ascii=True, indent=4)
    with open(targetDataDir + '/annotations_test.json', 'w', encoding='utf-8') as f:
        json.dump(test_json, f, ensure_ascii=True, indent=4)

    tmp_df = df.groupby('id').agg('first').reset_index()
    _ = Parallel(n_jobs=-1,
                 backend='threading')(delayed(run_copy)(row) for _, row in tqdm(
        tmp_df.iterrows(), total=len(tmp_df)))

    # check
    annFile = Path(targetDataDir + '/annotations_valid.json')
    coco = COCO(annFile)
    imgIds = coco.getImgIds()

    imgs = coco.loadImgs(imgIds[-3:])
    _, axs = plt.subplots(len(imgs), 2)
    for img, ax in zip(imgs, axs):
        I = Image.open(dataDir / img['file_name'])
        annIds = coco.getAnnIds(imgIds=[img['id']])
        anns = coco.loadAnns(annIds)
        ax[0].imshow(I)
        ax[1].imshow(I)
        plt.sca(ax[1])
        coco.showAnns(anns, draw_bbox=True)
    plt.tight_layout()
    plt.show()
