from pycocotools.coco import COCO
import pandas as pd
import cv2
import matplotlib.pyplot as plt

ROOT = '..'
DATA_DIR = './sartorius_coco_dataset'

if __name__ == "__main__":
    # Train Data
    df = pd.read_csv(f'{ROOT}/train.csv')
    df['image_path'] = ROOT + '/train/' + df['id'] + '.png'
    tmp_df = df.drop_duplicates(subset=["id", "image_path"]).reset_index(drop=True)
    tmp_df["annotation"] = df.groupby("id")["annotation"].agg(list).reset_index(drop=True)
    df = tmp_df.copy()
    df['label'] = df.cell_type.map({v: k for k, v in enumerate(df.cell_type.unique())})
    df['num_ins'] = df.annotation.map(lambda x: len(x))
    print(df.head(2))

    annFile = f'{DATA_DIR}/annotations_train.json'
    coco = COCO(annFile)
    imgIds = coco.getImgIds()

    tmp_df = df.query("num_ins<=30 and num_ins>=15").head(2)
    _, axs = plt.subplots(len(tmp_df), 2, figsize=(10, 5 * len(tmp_df)))
    for (_, row), ax in zip(tmp_df.iterrows(), axs):
        img = cv2.imread(DATA_DIR + f'/train/{row.id}.png')
        img_img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img[..., 0])
        annIds = coco.getAnnIds(imgIds=[row.id])
        anns = coco.loadAnns(annIds)
        ax[0].imshow(img)
        ax[1].imshow(img)
        plt.sca(ax[1])
        coco.showAnns(anns, draw_bbox=True)
    plt.tight_layout()
    plt.show()
