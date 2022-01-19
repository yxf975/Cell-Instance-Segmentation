import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import mmcv
import cv2
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv import Config

IMG_WIDTH = 704
IMG_HEIGHT = 520
confidence_thresholds = {0: 0.25, 1: 0.55, 2: 0.35}
pixel_thresholds = {0: 75, 1: 150, 2: 75}


def does_overlap(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            return True
    return False


def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            print("Overlap detected")
            mask[np.logical_and(mask, other_mask)] = 0
    return mask


def get_mask_from_result(result):
    d = {True: 1, False: 0}
    u, inv = np.unique(result, return_inverse=True)
    mk = cp.array([d[x] for x in u])[inv].reshape(result.shape)
    #     print(mk.shape)
    return mk


def get_img_and_mask(img_path, annotation, width, height):
    """ Capture the relevant image array as well as the image mask """
    img_mask = np.zeros((height, width), dtype=np.uint8)
    for i, annot in enumerate(annotation):
        img_mask = np.where(rle_decode(annot, (height, width)) != 0, i, img_mask)
    img = cv2.imread(img_path)[..., ::-1]
    return img[..., 0], img_mask


def plot_img_and_mask(img, mask, invert_img=True, boost_contrast=True):
    """ Function to take an image and the corresponding mask and plot

    Args:
        img (np.arr): 1 channel np arr representing the image of cellular structures
        mask (np.arr): 1 channel np arr representing the instance masks (incrementing by one)
        invert_img (bool, optional): Whether or not to invert the base image
        boost_contrast (bool, optional): Whether or not to boost contrast of the base image

    Returns:
        None; Plots the two arrays and overlays them to create a merged image
    """
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 3, 1)
    _img = np.tile(np.expand_dims(img, axis=-1), 3)

    # Flip black-->white ... white-->black
    if invert_img:
        _img = _img.max() - _img

    if boost_contrast:
        _img = np.asarray(ImageEnhance.Contrast(Image.fromarray(_img)).enhance(16))

    plt.imshow(_img)
    plt.axis(False)
    plt.title("Cell Image", fontweight="bold")

    plt.subplot(1, 3, 2)
    _mask = np.zeros_like(_img)
    _mask[..., 0] = mask
    plt.imshow(mask, cmap='rainbow')
    plt.axis(False)
    plt.title("Instance Segmentation Mask", fontweight="bold")

    merged = cv2.addWeighted(_img, 0.75, np.clip(_mask, 0, 1) * 255, 0.25, 0.0, )
    plt.subplot(1, 3, 3)
    plt.imshow(merged)
    plt.axis(False)
    plt.title("Cell Image w/ Instance Segmentation Mask Overlay", fontweight="bold")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cfg = Config.fromfile('./config_aug_exp.py')
    print(f'Config:\n{cfg.pretty_text}')
    ckpt = '../model/best_segm_mAP_epoch_14.pth'
    device = 'cuda:0'
    model = init_detector(config=cfg, checkpoint=ckpt, device=device)
    for file in sorted(os.listdir('../data/test')):
        img = mmcv.imread('../data/test/' + file)
        result = inference_detector(model, img)
        show_result_pyplot(model, img, result)
        all_masks = []
        for i, classe in enumerate(result[0]):
            if classe.shape != (0, 5):
                bbs = classe
                sgs = result[1][i]
                for bb, sg in zip(bbs, sgs):
                    #                 print(sg)
                    box = bb[:4]
                    cnf = bb[4]
                    count = np.count_nonzero(sg)
                    if cnf >= confidence_thresholds[i] and count >= pixel_thresholds[i]:
                        #                 if cnf >= confidence_thresholds[i]:
                        mask = get_mask_from_result(sg)
                        mask = remove_overlapping_pixels(mask, previous_masks)
                        all_masks.append(mask)

    # evaluation

