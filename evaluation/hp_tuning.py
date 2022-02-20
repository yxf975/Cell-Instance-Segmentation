import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import mmcv
import cv2
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv import Config
from pycocotools.coco import COCO
import cupy as cp
import argparse
import json
import os

WIDTH = 704
HEIGHT = 520
pixel_thresholds = {0: 75, 1: 150, 2: 75}
cell_type = ['shsy5y', 'astro', 'cort']


def does_overlap(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            return True
    return False


def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            # print("Overlap detected")
            mask[np.logical_and(mask, other_mask)] = 0
    return mask


def get_mask_from_result(result):
    d = {True: 1, False: 0}
    u, inv = np.unique(result, return_inverse=True)
    mk = cp.array([d[x] for x in u])[inv].reshape(result.shape)
    #     print(mk.shape)
    return mk


#
# def get_img_and_mask(img_path, annotation, width, height):
#     """ Capture the relevant image array as well as the image mask """
#     img_mask = np.zeros((height, width), dtype=np.uint8)
#     for i, annot in enumerate(annotation):
#         img_mask = np.where(rle_decode(annot, (height, width)) != 0, i, img_mask)
#     img = cv2.imread(img_path)[..., ::-1]
#     return img[..., 0], img_mask


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


# metric functions
def compute_iou(labels, y_pred):
    """
    Computes the IoU for instance labels and predictions.

    Args:
        labels (np array): Labels.
        y_pred (np array): predictions

    Returns:
        np array: IoU matrix, of size true_objects x pred_objects.
    """

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    # Compute intersection between all objects
    intersection = np.histogram2d(
        labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
    iou = intersection / union

    return iou[1:, 1:]  # exclude background


def precision_at(threshold, iou):
    """
    Computes the precision at a given threshold.

    Args:
        threshold (float): Threshold.
        iou (np array [n_truths x n_preds]): IoU matrix.

    Returns:
        int: Number of true positives,
        int: Number of false positives,
        int: Number of false negatives.
    """
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) >= 1  # Correct objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    return tp, fp, fn


def iou_map(truths, preds, verbose=0):
    """
    Computes the metric for the competition.
    Masks contain the segmented pixels where each object has one value associated,
    and 0 is the background.

    Args:
        truths (list of masks): Ground truths.
        preds (list of masks): Predictions.
        verbose (int, optional): Whether to print infos. Defaults to 0.

    Returns:
        float: mAP.
    """
    ious = [compute_iou(truth, pred) for truth, pred in zip(truths, preds)]

    #     print(ious[0].shape)

    if verbose:
        print("Thresh\tTP\tFP\tFN\tPrec.")

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tps, fps, fns = 0, 0, 0
        for iou in ious:
            tp, fp, fn = precision_at(t, iou)
            tps += tp
            fps += fp
            fns += fn

        p = tps / (tps + fps + fns)
        prec.append(p)

        if verbose:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tps, fps, fns, p))

    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

    return np.mean(prec)


if __name__ == "__main__":
    # parse para
    parser = argparse.ArgumentParser(description='evaluation of a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('model', help='trained model path')
    args = parser.parse_args()
    print(args.config)
    print(args.model)

    dir_path = os.path.dirname(args.model)
    print(dir_path)

    # cfg = Config.fromfile('../configs/config_aug_exp.py')
    cfg = Config.fromfile(args.config)
    print(f'Config:\n{cfg.pretty_text}')
    ckpt = args.model
    device = 'cuda:0'
    model = init_detector(config=cfg, checkpoint=ckpt, device=device)
    annfile = '../data/sartorius_coco_dataset/annotations_fp.json'
    testdir = '../data/sartorius_coco_dataset/test/'
    coco = COCO(annfile)
    print(coco.cats)
    score_collect = [[] for _ in range(3)]
    thres = np.linspace(0.0, 0.9, 19)
    print("---------------------evaluation-------------------------")
    for thre in thres:
        scores = [[], [], []]
        for imgid, imgInfo in coco.imgs.items():
            imgPath = testdir + imgInfo['file_name']
            annIds = coco.getAnnIds(imgIds=[imgid])
            anns = coco.loadAnns(annIds)
            cat = anns[0]['category_id']
            ann_mask = np.zeros((HEIGHT, WIDTH))
            for ann in anns:
                ann_mask = np.logical_or(ann_mask, coco.annToMask(ann))
            img = mmcv.imread(imgPath)
            result = inference_detector(model, img)
            model.eval()
            pred_class_ls = [len(result[0][0]), len(result[0][1]), len(result[0][2])]
            pred_class = pred_class_ls.index(max(len(result[0][0]), len(result[0][1]), len(result[0][2]))) + 1
            # print(cat, pred_class)
            if cat != pred_class:
                print("big big error-------------big big error-----------------big big error------")
                print("big big error-------------big big error-----------------big big error------")
                print("big big error-------------big big error-----------------big big error------")
            pred_mask = []
            pred = np.zeros((HEIGHT, WIDTH))
            for i, classe in enumerate(result[0]):
                if classe.shape != (0, 5):
                    bbs = classe
                    sgs = result[1][i]
                    for bb, sg in zip(bbs, sgs):
                        box = bb[:4]
                        cnf = bb[4]
                        count = np.count_nonzero(sg)
                        if cnf >= thre and count >= pixel_thresholds[i]:
                            #                 if cnf >= confidence_thresholds[i]:
                            mask = get_mask_from_result(sg)
                            mask = remove_overlapping_pixels(mask, pred_mask)
                            pred = np.logical_or(cp.asnumpy(mask), pred)
                            pred_mask.append(mask)
            scores[cat - 1].append(iou_map(ann_mask, pred))
        print("current thre:", thre)
        for i in range(3):
            score_collect[i].append(np.mean(scores[i]))
            print("mAP for class {} :".format(cell_type[i]), np.mean(scores[i]))
    print(thres)
    print(score_collect)
    confidence_thresholds = {}
    for i in range(3):
        ind = np.argmax(score_collect[i])
        confidence_thresholds[i] = thres[ind]
    print(confidence_thresholds)
    with open(dir_path+'/thres.json', 'w', encoding='utf-8') as f:
        json.dump(confidence_thresholds, f)

    # -----------------------infer情况----------------------------

    # for file in sorted(os.listdir('../data/test')):
    #     img = mmcv.imread('../data/test/' + file)
    #     result = inference_detector(model, img)
    #     show_result_pyplot(model, img, result)
    #     previous_masks = []
    #     for i, classe in enumerate(result[0]):
    #         if classe.shape != (0, 5):
    #             bbs = classe
    #             sgs = result[1][i]
    #             for bb, sg in zip(bbs, sgs):
    #                 #                 print(sg)
    #                 box = bb[:4]
    #                 cnf = bb[4]
    #                 count = np.count_nonzero(sg)
    #                 if cnf >= confidence_thresholds[i] and count >= pixel_thresholds[i]:
    #                     #                 if cnf >= confidence_thresholds[i]:
    #                     mask = get_mask_from_result(sg)
    #                     mask = remove_overlapping_pixels(mask, previous_masks)
    #                     previous_masks.append(mask)
