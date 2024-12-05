import numpy as np
from tqdm import tqdm
import PIL
from firelight.visualizers.colorization import get_distinct_colors
from cellpose import models, metrics
import skimage
from skimage.morphology import dilation, erosion, closing, ball, disk
from skimage.measure import regionprops
import os
import h5py
from tifffile import imread
from skimage import measure
import cv2
import matplotlib.pyplot as plt
import pickle
from pkg_resources import resource_stream


# load data path from file
def get_path():

    with resource_stream(__name__, os.path.join("..", "path")) as file:
        lines = file.readlines()

    lines = [line.decode('ascii').split(" ") for line in lines]
    path_dict = {line[0]: line[1].strip("\n") for line in lines}

    try:
        return path_dict["data_path"]
    except KeyError:
        print("There is no path 'data_path'.")


# image outputs
def alpha_blending(img1, img2, alpha=0.2):
    """
    Overlays img1 over img2 via alpha blending.
    :param img1: Image to overlay should be RGBA with floats
    :param img2: Background image, sould be RGBA with floats
    :param alpha: Blending value, float
    :return: Overlayed image in RGBA float
    """
    alpha1 = img1[:, :, -1][:, :, None] * alpha
    alpha2 = img2[:, :, -1][:, :, None]
    new_alpha = alpha1 + alpha2 * (1- alpha1)
    overlay = img1[:, : ,:-1] * alpha1 + img2[:, : ,:-1] * alpha2 *(1- alpha1)
    overlay = overlay / new_alpha
    overlay  = np.concatenate((overlay, new_alpha), axis = -1)
    return overlay


def rescale_raw_to_rgba(image):
    """ Rescales grayscale to [0, 1] and converts to rgba"""
    image = image - image.min()
    image = image / image.max()

    img_rgb = np.tile(image[:, :, None], (1,1,3))

    img_rgba = np.concatenate([img_rgb, np.ones((*img_rgb.shape[:2], 1))], axis=-1)
    return img_rgba

def get_rgba_from_label_mask(mask, seed=0, n_colors=None):
    """ Turn a interger mask into one colored by rgba values, with alpha channedl
    indicating objects."""
    np.random.seed(seed)
    if n_colors is None:
        n_colors = np.maximum(len(np.unique(mask))-1, 1) # prevents error in case of empty mask
    colors = get_distinct_colors(n_colors)
    np.random.shuffle(colors)
    colors = np.concatenate([np.zeros((1,3)), colors], axis=0)  # add black for background

    mask_rgb = colors[mask.reshape(-1)].reshape(*mask.shape, 3)
    alpha_mask = (mask > 0).astype(float)
    mask_rgba = np.concatenate((mask_rgb, alpha_mask[:, :, None]), axis=-1)
    return mask_rgba

def get_rgba_from_rgb(img):
    """Turn RGB image into RGBA image with alpha channel indicating non-black
    pixels"""
    assert len(img.shape) == 3 and img.shape[2] == 3
    alpha_mask = np.any(img, axis=2)
    img_rgba = np.concatenate((img / 255, alpha_mask[:, :, None]), axis=-1)
    return img_rgba

def overlay_img_with_mask(img, mask, alpha=0.3, seed=0, n_colors=None):
    """Overlay an image with a segmentation mask."""
    img = rescale_raw_to_rgba(img)
    mask = get_rgba_from_label_mask(mask, seed, n_colors=n_colors)
    overlay = alpha_blending(mask, img, alpha)
    return overlay


def save_overlays(images, seg, shape, path_to_file, n_colors=None):
    overlays = []
    for i in tqdm(range(len(images))):
        overlay = (255 * overlay_img_with_mask(images[i], seg[i].astype(int), n_colors=n_colors)).astype(np.uint8)
        overlays.append(overlay)
    overlays = np.stack(overlays)
    overlays = overlays.reshape(*shape, 4)[:, :, None]

    skimage.io.imsave(path_to_file+".tiff", overlays, imagej=True)


def tif_to_h5(data_dir):
    # load all tif files in the directory and turn them into .h5 files
    for _, _, file_names in os.walk(data_dir):
        for file_name in file_names:
            if file_name.endswith(".tif"):
                img = imread(os.path.join(data_dir, file_name))
                with h5py.File(os.path.join(data_dir, file_name.removesuffix(
                        ".tif") + ".h5"), "w") as file:
                    if img.shape[1] == 2:
                        # labelled data, no time
                        file.create_dataset("data", data=img[:, 0])
                        file.create_dataset("seg", data=img[:, 1])
                    else:
                        # dims are time, z, x, y
                        file.create_dataset("data", data=img[:])


# load predictions of a dataset with a trained model
def load_predictions(path, dataset, file_names, exp_name):
    exp_path = os.path.join(path, dataset, "results", exp_name)
    preds = []
    for file_name in file_names:
        file_name = f"{file_name}_{exp_name}.h5"
        with h5py.File(os.path.join(exp_path, file_name), "r") as file:
            pred = file["seg"][:]
        preds.append(pred)
    return np.stack(preds)


# load the test set part of a CV experiment from all folds, so that every image appears
def load_test_predictions(path, dataset, file_names, exp_base_name, seed=0, n_folds=10):
    preds_unordered = []
    all_test_idx = []
    for fold in range(n_folds):
        # get test idx for this fold
        _, test_idx = train_test_idx(len(file_names), seed=seed, fold=fold, n_folds=n_folds)
        all_test_idx.extend(test_idx)

        # create experiment name including the fold
        exp_name_suffix = "_".join(exp_base_name.split("_")[-2:]) # remove stitch or aniso part
        if "epoch" in exp_base_name:
            ind = exp_base_name.split("_").index("epoch")
            epoch = exp_base_name.split("_")[ind +1]
            exp_name_suffix = f"epoch_{epoch}_{exp_name_suffix}"
        exp_name = exp_base_name.removesuffix(exp_name_suffix)
        exp_name = f"{exp_name}fold_{fold}_of_{n_folds}_{exp_name_suffix}"

        # load the test predictions of this fold
        preds_fold = load_predictions(path, dataset, file_names[test_idx], exp_name)
        preds_unordered.extend(preds_fold)

    all_test_idx = np.array(all_test_idx)

    return np.stack(preds_unordered), all_test_idx


# aggressive merging strategy that iteratively merges segments with every small segment it neighbors
def handle_small_segments_old(seg, min_size, nbhd_size=1):
    seg = seg.copy()
    # find segment sizes
    seg_ids, seg_sizes = np.unique(seg, return_counts=True)

    assert seg_ids.max() == len(
        seg_ids) - 1, f"Highest label is {seg_ids.max()} but there are only {len(seg_ids) - 1} segments."

    # sort by size to make order independent of arbitrary label assignment
    perm = np.argsort(seg_sizes)
    seg_ids_sorted = seg_ids[perm]
    seg_sizes_sorted = seg_sizes[perm]

    small_seg_props = {}

    for id, size in zip(seg_ids_sorted, seg_sizes_sorted):
        # do not merge large segments or background
        if size >= min_size or id == 0: continue

        # find all neighbors within nbhd_radius
        seg_mask = (seg == id)
        # only do dilation locally in extend bounding box
        slice_z, slice_x, slice_y = get_bbox_margin(seg_mask, nbhd_size)
        seg_mask_bbox = seg_mask[slice_z, slice_x, slice_y]

        dilated_seg_mask_bbox = dilation(seg_mask_bbox, ball(nbhd_size))
        nbhd_mask_bbox = (
                    dilated_seg_mask_bbox.astype(int) - seg_mask_bbox.astype(
                int)).astype(bool)

        nb_ids, _ = np.unique(
            seg[slice_z, slice_x, slice_y][nbhd_mask_bbox], return_counts=True)


        non_bg = nb_ids != 0
        nb_ids = nb_ids[non_bg]  # filter background from neighbor list

        # if segment has no neighbors, set it to background already, since it is small
        if len(nb_ids) == 0:
            seg[seg == id] = 0

        nb_ids = np.append(nb_ids,
                           [id])  # add self as neighbor

        props = {"nb_ids": nb_ids}

        small_seg_props[id] = props

    n_small_segments = ((seg_sizes < min_size) * (seg_ids != 0)).sum()

    # reassign small segments to larger segments n_small_segments many times,
    # so that even the longest chain of small segments gets merged
    for i in range(n_small_segments):
        # loop over small segemnts
        for id in list(small_seg_props.keys()):
            # find largest neighbor
            nb_ids = small_seg_props[id]["nb_ids"]
            nb_sizes = seg_sizes[nb_ids]
            id_big_brother = nb_ids[np.argmax(nb_sizes)]

            # do not merge locally largest segement to smaller one
            if id_big_brother == id:
                continue

            # assign id to id_big_brother, update sizes and remove id from dict
            seg_sizes[id_big_brother] += seg_sizes[id]
            seg[seg == id] = id_big_brother
            del small_seg_props[id]

            # update the dict
            for other_id in small_seg_props:
                nb_list = small_seg_props[other_id]["nb_ids"]
                nb_list[nb_list == id] = id_big_brother

    # remove segments that are still too small
    for id in small_seg_props:
        if seg_sizes[id] < min_size:
            seg[seg == id] = 0

    # relabel, so that all labels are contiguous
    seg = relabel(seg)
    return seg


# delete all segments that touch the volume boundary
def delete_boundary_segments(seg, exclude_id=None):
    # delete boundary segments
    boundary_xy = np.zeros_like(seg)
    boundary_xy[:, 0] = 1
    boundary_xy[:, -1] = 1
    boundary_xy[:, :, 0] = 1
    boundary_xy[:, :, -1] = 1

    boundary_segs = np.unique(seg[boundary_xy.astype(bool)])


    for id in boundary_segs:
        if id == exclude_id: continue
        seg[seg == id] = 0
    return seg


# select the segment that occupies most of the central area of a volume as central segment
def find_central_id(seg, xy_margin=30, min_size=10000):
    seg_central = seg[:, xy_margin:-xy_margin, xy_margin:-xy_margin]

    ids, sizes = np.unique(seg_central, return_counts=True)

    if len(sizes) == 1 or max(sizes[1:]) < min_size:
        return None

    id_main = ids[1:][np.argmax(sizes[1:])]

    return id_main


def merge_allowed(main_mask, nb_mask, threshold=0.005):
    # forbid a merge if it makes the segment muss less convex

    merged_mask = np.logical_or(main_mask, nb_mask)

    # compute convexity of to-merge segments
    merged_props = regionprops(merged_mask.astype(int))[0]
    merged_ratio = merged_props.area / merged_props.area_convex

    # compute convexity of main segment
    props = regionprops(main_mask.astype(int))[0]
    ratio = props.area / props.area_convex

    # only allow a merge that make the merged not much loss convex than before
    rel_change = (merged_ratio - ratio) / ratio
    return rel_change > -threshold

def get_nbs(seg, seg_mask,  nbhd_size=1):
    # only do dilation locally in extend bounding box
    slice_z, slice_x, slice_y = get_bbox_margin(seg_mask, nbhd_size)
    seg_mask_bbox = seg_mask[slice_z, slice_x, slice_y]

    dilated_seg_mask_bbox = dilation(seg_mask_bbox, ball(nbhd_size))
    nbhd_mask_bbox = (dilated_seg_mask_bbox.astype(int) - seg_mask_bbox.astype(
        int)).astype(bool)

    nb_ids, contact_nb = np.unique(
        seg[slice_z, slice_x, slice_y][nbhd_mask_bbox],
        return_counts=True)
    non_bg = nb_ids != 0
    return nb_ids[non_bg], contact_nb[non_bg]


# find central segment and add connected segments based on convexity and contact threshold
def handle_small_segments_simple(seg, nbhd_size=1, xy_margin=30, threshold=0.005, contact_threshold=450):
    seg = seg.copy()

    id_main = find_central_id(seg, xy_margin=xy_margin)

    if id_main is None:
        return np.zeros_like(seg)

    # get segments and their neighborhoods
    seg_ids, seg_sizes = np.unique(seg, return_counts=True)

    # omit background
    seg_ids = seg_ids[1:]
    seg_sizes = seg_sizes[1:]

    if len(seg_sizes) == 0:
        return seg

    seg_props = {}
    for ind in seg_ids:
        # find all neighbors within nbhd_size
        seg_mask = (seg == ind)
        nb_ids, contact = get_nbs(seg, seg_mask, nbhd_size)
        seg_props[ind] = {"nb_ids": nb_ids,
                         "contact": contact}

    # merge neighbors of main object iteratively
    has_changed = True  # records if there has been a change since last considering the neighbor

    while len(seg_props[id_main]["nb_ids"]) > 0:
        # merge one neighbor
        nb_id = seg_props[id_main]["nb_ids"][0]
        contact = seg_props[id_main]["contact"][0]

        # check if merge is ok, only merge if not less convex or if large contact area
        if merge_allowed((seg == id_main).astype(int),
                         (seg == nb_id).astype(int),
                         threshold) or contact > contact_threshold:

            # perform the merge and do some bookkeeping about new neighbors and contact areas
            seg[seg == nb_id] = id_main

            # update the neighbors of main
            seg_props[id_main]["nb_ids"] = np.append(seg_props[id_main]["nb_ids"], seg_props[nb_id]["nb_ids"])
            seg_props[id_main]["contact"] = np.append(seg_props[id_main]["contact"], seg_props[nb_id]["contact"])

            # filer id_main and nb_id from neighbor list
            not_main_id = seg_props[id_main]["nb_ids"] != id_main
            not_nb_id = seg_props[id_main]["nb_ids"] != nb_id
            mask_true_nbs = np.logical_and(not_main_id, not_nb_id)

            seg_props[id_main]["nb_ids"] = \
                seg_props[id_main]["nb_ids"][mask_true_nbs]

            seg_props[id_main]["contact"] = \
                seg_props[id_main]["contact"][mask_true_nbs]

            for i in seg_props:
                seg_props[i]["nb_ids"][seg_props[i]["nb_ids"] == nb_id] = id_main
            has_changed = True

        else:
            # delete neighbor from main cell's neighbor list if considered again after no change has occurred, otherwise roll to back
            if has_changed:
                seg_props[id_main]["nb_ids"] = np.roll(seg_props[id_main]["nb_ids"], -1)
                seg_props[id_main]["contact"] = np.roll(seg_props[id_main]["contact"], -1)

            else:
                seg_props[id_main]["nb_ids"] = seg_props[id_main]["nb_ids"][1:]
                seg_props[id_main]["contact"] = seg_props[id_main]["contact"][1:]

            has_changed = False

    # delete all unmerged segments but main
    seg[seg != id_main] = 0

    return seg


# this function is deprecated, do not use
def handle_small_segments(seg, min_size, nbhd_size=1, contact_thres=0.1):
    seg = seg.copy()
    # find segment sizes
    seg_ids, seg_sizes = np.unique(seg, return_counts=True)

    assert seg_ids.max() == len(
        seg_ids) - 1, f"Highest label is {seg_ids.max()} but there are only {len(seg_ids) - 1} segments."

    # sort by size to make independent of arbitrary label assignment
    perm = np.argsort(seg_sizes)
    seg_ids_sorted = seg_ids[perm]
    seg_sizes_sorted = seg_sizes[perm]

    small_seg_props = {}

    for id, size in zip(seg_ids_sorted, seg_sizes_sorted):
        if size >= min_size or id == 0: continue

        # find all neighbors within nbhd_radius
        seg_mask = (seg == id)
        # only do dilation locally in extend bounding box
        slice_z, slice_x, slice_y = get_bbox_margin(seg_mask, nbhd_size)
        seg_mask_bbox = seg_mask[slice_z, slice_x, slice_y]

        dilated_seg_mask_bbox = dilation(seg_mask_bbox, ball(nbhd_size))
        nbhd_mask_bbox = (
                    dilated_seg_mask_bbox.astype(int) - seg_mask_bbox.astype(
                int)).astype(bool)

        nb_ids, contact_nb = np.unique(
            seg[slice_z, slice_x, slice_y][nbhd_mask_bbox], return_counts=True)

        print(id)
        print(contact[1:])
        print(nbhd_mask_bbox.sum())
        print("\n")
        non_bg = nb_ids != 0
        nb_ids = nb_ids[non_bg]  # filter background from neighbor list
        contact_nb = contact_nb[non_bg]

        contact = np.zeros(len(seg_ids))
        contact[nb_ids] = contact_nb

        # if segment has no neighbors, set it to background already
        if len(nb_ids) == 0:
            seg[seg == id] = 0

        nb_ids = np.append(nb_ids,
                           [id])  # add self as neighbor

        props = {"nb_ids": nb_ids,
                 "nbhd_size": nbhd_mask_bbox.sum(),
                 "contact": contact}

        small_seg_props[id] = props

    n_small_segments = ((seg_sizes < min_size) * (seg_ids != 0)).sum()

    # reassign small segments to larger segments n_small_segments many times,
    # so that even the longest chain of small segments gets merged
    for i in range(n_small_segments):
        for id in small_seg_props:
            # merge with segment that has largest contact area, if above threshold
            contact_shares = small_seg_props[id]["contact"] / \
                             small_seg_props[id]["nbhd_size"]
            nb_ids = small_seg_props[id]["nb_ids"]

            # find neighbor with largest shared boundary, if above threshold
            mask_valid_contacts = contact_shares >= contact_thres
            if mask_valid_contacts.sum() == 0: continue
            id_best_neighbor = nb_ids[mask_valid_contacts][
                np.argmax(contact_shares[mask_valid_contacts])]

            # merge to id_best_neighbor, update size, neighbors and contact areas
            seg_sizes[id_best_neighbor] += seg_sizes[id]
            seg[seg == id] = id_best_neighbor
            for id_other_nb in nb_ids:
                continue
            #contact[]

            del small_seg_props[id]

            nb_ids = small_seg_props[id]["nb_ids"]
            nb_sizes = seg_sizes[nb_ids]
            id_big_brother = nb_ids[np.argmax(nb_sizes)]

            if id_big_brother == id: continue

            # assign id to id_big_brother, update sizes and remove id from dict
            seg_sizes[id_big_brother] += seg_sizes[id]
            seg[seg == id] = id_big_brother
            del small_seg_props[id]

            # update the dict
            for id in small_seg_props:
                nb_list = small_seg_props[id]["nb_list"]
                nb_list[nb_list == id] = id_big_brother

    # remove segments that are still too small
    for id in small_seg_props:
        if seg_sizes[id] < min_size:
            seg[seg == id] = 0

    # relabel, so that all labels are contiguous
    seg = relabel(seg)

    return seg


# merge all parasite segments inside each RBC
def merge_in_rbc(para_seg, rbc_pp_seg):
    new_para_seg = []
    for p_seg, r_seg in zip(para_seg, rbc_pp_seg):
        r_mask = r_seg > 0

        para_in_rbc = p_seg[r_mask]
        new_seg = np.zeros_like(p_seg)
        new_seg[r_mask] = para_in_rbc > 0
        new_para_seg.append(new_seg)
    new_para_seg = np.stack(new_para_seg)
    return new_para_seg


# relabel a segmentation so that it uses labels 0, 1, ..., max_segement
def relabel(seg):
    for i, label in enumerate(np.unique(seg)):
        seg[seg == label] = i
    return seg


# smooth segements
def smooth_segments(seg, radius=1, do_3d=True):
    labels, counts = np.unique(seg, return_counts=True)

    # resort to process largest segment, which is probably the central object, last
    perm = np.argsort(counts)[::-1]
    labels = labels[perm]
    counts = counts[perm]

    new_seg = np.copy(seg)
    for label, count in zip(labels, counts):
        # do not smooth the background
        if label == 0:
            continue

        # smooth inside a bounding box for performance reasons
        mask = seg == label
        slice_z, slice_x, slice_y = get_bbox_margin(mask, margin=radius)

        mask_bbox = mask[slice_z, slice_x, slice_y]

        if do_3d:
            # dilate in 3D, but only erode in 2D
            if radius > 0:
                struc_elem_3D = ball(radius)
                struc_elem_3D = np.stack([struc_elem_3D[i] for i in [0, radius+1, -1]])
            else:
                struc_elem_3D = np.array([1,1,1])[:, None, None]
            mask_bbox = dilation(mask_bbox, struc_elem_3D)

            struc_elem = disk(radius)
            closed_mask_bbox = np.stack([erosion(mask_slice, struc_elem) for mask_slice in mask_bbox])
        else:
            #*close* in 2D.
            struc_elem = disk(radius)
            closed_mask_bbox = np.stack([closing(mask_slice, struc_elem) for mask_slice in mask_bbox])

        new_seg[slice_z, slice_x, slice_y][closed_mask_bbox] = label

    return new_seg


# faster bounding boxes slices than with scipy.ndimage.find_objects
def get_bbox_slices(mask):
    """
    Finds bounding box slices for a binary mask containing a single connected object
    :param mask: binary mask with a single object
    :return: slice_x, slice_y, slice_z: slices of the object in x any y direction
    """
    ndims = len(mask.shape)
    slices = []
    for axis, shape in enumerate(mask.shape):
        coords = np.arange(0, shape, dtype=int)
        other_axes = tuple((np.arange(ndims-1) + axis +1 ) % ndims )
        shadow = mask.max(other_axes) # take maximum over all other axes
        obj_coords = coords[shadow]
        slice_axis = slice(obj_coords.min(), obj_coords.max()+1)
        slices.append(slice_axis)
    return tuple(slices)


# gets bounding box of mask with a given margin
def get_bbox_margin(mask, margin):
    # extract bounding box of object and extend it by margin
    ndims = len(mask.shape)
    slices = get_bbox_slices(mask.astype(bool))
    assert len(slices) == ndims

    if isinstance(margin, int):
        margins = np.ones(ndims) * margin # use same margin in all axes
    else:
        assert len(margin) == ndims
        margins = margin

    new_slices = []
    for i, (slice_i, margin_i) in enumerate(zip(slices, margins)):
        slice_start = max(slice_i.start - margin_i, 0)
        slice_stop = min(slice_i.stop + margin_i, mask.shape[i])
        new_slices.append(slice(int(slice_start), int(slice_stop)))
    return tuple(new_slices)

def get_shell(seg_mask, erosion_radius=1, dilation_radius=1, anisotropy=3.2):

    assert dilation_radius >= -erosion_radius
    seg_mask = seg_mask.astype(bool)

    dilation_xy = dilation_radius * int(anisotropy)
    erosion_xy = erosion_radius * int(anisotropy)


    net_dilation = np.max(dilation_radius, 0)
    net_dilation_xy = net_dilation * int(anisotropy)
    slice_z, slice_x, slice_y = get_bbox_margin(seg_mask, margin=[net_dilation,
                                                                  net_dilation_xy,
                                                                  net_dilation_xy])
    seg_bbox = seg_mask[slice_z, slice_x, slice_y]

    # negative dilation is treated as erosion and vice versa
    if dilation_radius >= 0:
        dilated_mask_bbox = dilation(seg_bbox, ball(dilation_xy)[::int(anisotropy)])
    else:
        dilated_mask_bbox = erosion(seg_bbox, ball(-dilation_xy)[::int(anisotropy)])

    if erosion_radius >= 0:
        eroded_mask_bbox = erosion(seg_bbox, ball(erosion_xy)[::int(anisotropy)])
    else:
        eroded_mask_bbox = dilation(seg_bbox, ball(-erosion_xy)[::int(anisotropy)])

    assert np.all(dilated_mask_bbox >= eroded_mask_bbox)

    shell_mask_bbox = (dilated_mask_bbox.astype(int) - eroded_mask_bbox.astype(int)).astype(bool)

    shell_mask = np.zeros_like(seg_mask)
    shell_mask[slice_z, slice_x, slice_y] = shell_mask_bbox

    return shell_mask


# compute the shells of all segments, width given by dilation_radius - erosion_radius
def get_shells(seg, erosion_radius=1, dilation_radius=1):

    labels, sizes = np.unique(seg, return_counts=True)

    # resort, so that shells of larger cells overlay shells of smaller cells
    perm = np.argsort(sizes)
    labels = labels[perm]

    shell_img = np.zeros_like(seg)

    for label in labels:
        if label == 0:
            continue  # no shell for background
        seg_mask = seg == label
        shell_mask = get_shell(seg_mask, erosion_radius, dilation_radius)
        shell_img[shell_mask] = label

    return shell_img

def compute_auc(data_dict, datasets=None, stage="pred_3D", gt_stage="gt"):
    # compute the area under the average precision curve. This is the final metric.
    all_metrics = []

    if datasets is None:
        datasets = data_dict.keys()

    for dataset in datasets:
        gts = data_dict[dataset][gt_stage]
        preds = data_dict[dataset][stage]

        dataset_metrics = []

        # compute the metrics
        for i, (gt, pred) in enumerate(zip(gts, preds)):
            if gt.max() == 0 and pred.max() == 0:
                continue  # no segments, so aps ill-defined and tp, fp, fn count not affected
            aps, tps, fps, fns, _ = get_metrics([gt],
                                                [pred])
            dataset_metrics.append(np.concatenate([aps, tps, fps, fns]))
        all_metrics.append(np.stack(dataset_metrics))

    # average over all images
    all_metrics = np.concatenate(all_metrics)
    mean_aps = all_metrics[:, 1] / (all_metrics[:, 1] + all_metrics[:, 2] + all_metrics[:, 3])
    mean_aps = mean_aps.mean(0)

    # compute approximate area
    return mean_aps.sum() / 11

def aps_by_train_test(cv_metrics, train=True, seed=0, n_folds=10):
    # separate by train and test
    cv_aps = []
    for metrics_by_fold in cv_metrics:
        aps_by_fold = []
        for i, fold in enumerate(metrics_by_fold):
            metrics = []
            # datasets need to be processed separately bc train / test split depends on size of dataset
            for dataset_metrics in fold:
                n_imgs = len(dataset_metrics["metrics"])
                train_idx, test_idx = train_test_idx(n_imgs, seed, i,
                                                     n_folds)
                if train:
                    metrics.append(dataset_metrics["metrics"][train_idx, 1:])
                else:
                    metrics.append(dataset_metrics["metrics"][test_idx, 1:])

            # flatten across the different datasets in the train and test sets
            metrics = np.stack([metric_of_image
                                for metrics_of_dataset in
                                metrics
                                for metric_of_image in
                                metrics_of_dataset])

            # compute mean aps
            aps_by_fold.append(mean_aps(metrics))

        cv_aps.append(np.stack(aps_by_fold))
    return cv_aps


def plot_mean_aps(ax, cv_aps, cmap, cv_model_names, train=True, thresholds=np.arange(11) / 10):
    if not train:
        best_cv_test_ind = 0
        best_cv_test_auc = 0


    for j in range(len(cv_aps)):
        aps_mean = cv_aps[j].mean(0)
        aps_std = cv_aps[j].std(0)


        if train:
            label = f"train_{cv_model_names[j]}"
            color_shift = 0
        else:
            label = f"test_{cv_model_names[j]}"
            color_shift = 1



        ax.plot(thresholds, aps_mean, c=cmap(2 * j + color_shift),
                 label=label)
        ax.fill_between(thresholds,
                         aps_mean - aps_std,
                         aps_mean + aps_std,
                         alpha=0.1,
                         color=cmap(2 * j + color_shift)
                         )

        # check best test aps of all models
        if not train and aps_mean.sum(-1) > best_cv_test_auc:
            best_cv_test_auc = aps_mean.sum(-1)
            best_cv_test_ind = j

    if train:
        return ax
    else:
        return ax, best_cv_test_ind, best_cv_test_auc


# print mean-AP curves for different models trained with crossvalidation
def train_test_auc(cv_model_names,
                   target_dir,
                   dataset,
                   thresholds=np.arange(11) / 10,
                   cmap_name="tab20",
                   n_folds=10,
                   seed=0,
                   stitch_threshold=0.1,
                   anisotropy=3.2,
                   ax=None,
                   return_metrics=False,
                   shells=False):

    # load pre-computed metrics
    cv_metrics = load_cv_metrics(target_dir,
                                 dataset,
                                 cv_model_names,
                                 n_folds=n_folds,
                                 stitch_threshold=stitch_threshold,
                                 anisotropy=anisotropy,
                                 shells=shells)

    # separate by train and test
    cv_aps_train = aps_by_train_test(cv_metrics,
                                     train=True,
                                     seed=seed,
                                     n_folds=n_folds)

    # there is only a test set if there are at least 2 folds
    if n_folds > 1:
        cv_aps_test = aps_by_train_test(cv_metrics,
                                     train=False,
                                     seed=seed,
                                     n_folds=n_folds)

    # plot everything
    cmap = plt.get_cmap(cmap_name)

    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax = plot_mean_aps(ax,
                       cv_aps_train,
                       cmap=cmap,
                       cv_model_names=cv_model_names,
                       train=True,
                       thresholds=thresholds)

    if n_folds > 1:
        ax, best_cv_test_ind, best_cv_test_auc \
            = plot_mean_aps(ax,
                            cv_aps_test,
                            cmap=cmap,
                            cv_model_names=cv_model_names,
                            train=False,
                            thresholds=thresholds)
        print(f"Best method: {cv_model_names[best_cv_test_ind]}")
        print(f"Best AUC: {best_cv_test_auc / len(thresholds)}")

    plt.legend(loc=(1., 0))

    if return_metrics:
        return ax, cv_aps_train, cv_aps_test

    return ax


def get_cv_metric_file_names(cv_model_name, n_folds=10, stitch_threshold=0.1, anisotropy=3.2, epoch=499, shells=False):
    # create list of model names with fold for the cv_model_name
    cv_metric_file_names = []
    epoch_str = f"_epoch_{epoch}"
    for j in range(n_folds):
        eval_str = cv_model_name.removesuffix(epoch_str)
        eval_str = eval_str + f"_fold_{j}_of_{n_folds}"
        eval_str = eval_str + epoch_str
        eval_str = get_eval_string(eval_str, stitch_threshold, anisotropy)
        if shells:
            eval_str = eval_str + "_shells"
        cv_metric_file_names.append(f"metrics_{eval_str}.pkl")
    return cv_metric_file_names


def get_eval_string(model_name, stitch_threshold=0.1, anisotropy=3.2):
    # extract substring used for segmentations and metric files
    eval_str = model_name.removeprefix("cellpose_")
    if "2D" in model_name:
        eval_str += f"_stitch_{stitch_threshold}"
    if "3D_iso" in model_name:
        eval_str += f"_aniso_{anisotropy}"
    return eval_str


# read out training datasets from model name
def datasets_from_model(model_name):
    l = model_name.split("_")
    ind1 = l.index("data")
    ind2 = l.index("by")
    datasets = l[ind1+1: ind2]
    return datasets


# load precomputed metrics for whole cross-val experiments
def load_cv_metrics(target_dir,
                    dataset,
                    cv_model_names,
                    n_folds=10,
                    stitch_threshold=0.1,
                    anisotropy=3.2,
                    epoch=499,
                    shells=False):
    cv_metrics = []
    for cv_model_name in cv_model_names:
        # get list of metric file names by fold
        fold_names = get_cv_metric_file_names(cv_model_name, n_folds, stitch_threshold, anisotropy, epoch, shells=shells)
        # load the metrics
        cv_metrics.append(load_metrics(target_dir, dataset, fold_names))
    return cv_metrics

# load pre-computed metrics for several datasets
def load_metrics(target_dir, dataset, metric_file_names):
    if type(dataset) is str:
        datasets = [dataset]
    else:
        datasets = dataset

    # load metrics for all datasets
    metrics = []
    for metric_file_name in metric_file_names:
        metrics_by_model = []
        exp_name = metric_file_name.removesuffix(".pkl").removeprefix("metrics_")
        if "shells" in exp_name:
            exp_name = exp_name.removesuffix("_shells")
        for dataset in datasets:
            with open(os.path.join(target_dir, dataset, "results", exp_name, metric_file_name), 'rb') as handle:
                metrics_by_model.append(pickle.load(handle))
        metrics.append(metrics_by_model)
    return metrics


# wrapper for obtaining data either by loading or from pre-loaded dict, chunks allow fast parallel loading
def get_data(dataset, datasets, data_path, chunk=0, n_chunks=1):
    if isinstance(datasets, list):
        i, s, fn = load_data(data_path, dataset)
    elif isinstance(datasets, dict):
        i = datasets[dataset]["imgs"]
        s = datasets[dataset]["segs"]
        fn = datasets[dataset]["file_names"]
    else:
        raise NotImplementedError

    # only return the required chunk, chunks only differ in size by at most 1
    l = len(i)
    min_chunk_size = l // n_chunks
    remainder = l - min_chunk_size * n_chunks
    chunk_sizes = np.ones(n_chunks).astype(int) * min_chunk_size
    chunk_sizes[0:remainder] += 1
    starts = np.concatenate([np.zeros(1).astype(int), np.cumsum(chunk_sizes)])

    start = starts[chunk]
    end = starts[chunk + 1]
    return i[start:end], s[start:end], fn[start:end]


# data loading
def load_data(path, dataset):
    data_dir = os.path.join(path, dataset, "data")

    # turn .tif into .h5
    tif_to_h5(data_dir)

    # load data
    imgs = []
    segs = []
    file_names = []

    # sort data dir contents to ensure correct reproduction of train / test split
    l = os.listdir(data_dir)
    l.sort()
    for file_name in l:
        if file_name.endswith(".h5"):
            with h5py.File(os.path.join(data_dir, file_name), "r") as file:
                img = file["data"][:]

                # if there is no segmentation, use empty segmentation as dummy
                if "seg" in file.keys():
                    seg = file["seg"][:]
                else:
                    seg = np.zeros_like(img)

            segs.append(seg)
            imgs.append(img)
            file_names.append(file_name.removesuffix(".h5"))
    imgs = np.stack(imgs)
    segs = np.stack(segs)
    file_names = np.stack(file_names)


    # relabel images with different but touching segments into a single segment (only implemented for volumes)
    if len(segs.shape) == 4:
        segs = [measure.label(seg, background=0, connectivity=2) for seg in
                     segs]
        segs = np.stack(segs).astype("uint16")
    return imgs, segs, file_names




def slice_by_mode(imgs, segs, file_names, mode="2D", anisotropy=3.2):
    # get list of image slices needed for cellpose training depending on training mode

    # in 2D training, the volumes need to be split into 2D slices
    if mode == "2D":
        n_stacks = imgs.shape[0]
        height = imgs.shape[1]
        img_shape = imgs.shape[-2:]
        imgs = imgs.reshape(-1, *img_shape)
        segs = segs.reshape(-1, *img_shape)
        # add the z-slice to the file names
        file_names= np.char.add(file_names.repeat(height),
                                    np.char.add(
                                        np.array(["_z_"] * height * n_stacks),
                                        np.tile(np.arange(height).astype(
                                            str), n_stacks)))

    elif mode == "3D":
        imgs, file_names = get_3D_slices(imgs, file_names)
        segs = get_3D_slices(segs)

    elif mode == "3D_iso":
        imgs, file_names = get_3D_slices(imgs, file_names, anisotropy=anisotropy)
        segs = get_3D_slices(segs, anisotropy=anisotropy)
    else:
        raise NotImplementedError(f"Mode must be '2D', '3D' or '3D_iso', but was {mode}")


    # filter masks of a single pixel as this creates an error with cellpose
    segs = [seg if (seg > 0).sum() > 1 else np.zeros_like(seg)
                     for seg in segs]

    return list(imgs), list(segs), list(file_names)


def get_3D_slices(imgs, file_names=None, anisotropy=False):
    # slices a 3D volume into three stacks of xy, zx, and zy slices, so that cellpose trains on all directions
    # assumes imgs of shape n_stacks * z * x * y
    n_stacks = imgs.shape[0]
    x_dim = imgs.shape[-2]
    y_dim = imgs.shape[-1]
    z_dim = imgs.shape[-3]


    imgs_xy = imgs.copy().reshape(-1, x_dim, y_dim)
    imgs_zx = np.moveaxis(imgs.copy(), -1, 1).reshape(-1, z_dim, x_dim)
    imgs_zy = np.moveaxis(imgs.copy(), -2, 1).reshape(-1, z_dim, y_dim)

    if anisotropy:
        imgs_zx = [cv2.resize(img,
                              (x_dim, int(z_dim * anisotropy)),  # order is swapped in the cv2 api
                              interpolation=cv2.INTER_LINEAR)
                   for img in imgs_zx]
        imgs_zy = [cv2.resize(img,
                              (y_dim, int(z_dim * anisotropy)),  # order is swapped in the cv2 api
                              interpolation=cv2.INTER_LINEAR)
                   for img in imgs_zy]
    imgs = list(imgs_xy) + list(imgs_zx) + list(imgs_zy)

    if file_names is not None:
        file_names_xy = np.char.add(file_names.repeat(z_dim),
                                    np.char.add(np.array(["_z_"] * z_dim * n_stacks),
                                                np.tile(np.arange(z_dim).astype(str),
                                                        n_stacks)))
        file_names_zx = np.char.add(file_names.repeat(y_dim),
                                    np.char.add(np.array(["_y_"] * y_dim * n_stacks),
                                                np.tile(np.arange(y_dim).astype(str),
                                                        n_stacks)))
        file_names_zy = np.char.add(file_names.repeat(x_dim),
                                    np.char.add(np.array(["_x_"] * x_dim * n_stacks),
                                                np.tile(np.arange(x_dim).astype(str),
                                                        n_stacks)))
        file_names = list(file_names_xy) + list(file_names_zx) + list(file_names_zy)
        return imgs, file_names

    return imgs


def train_test_idx(n, seed, fold, n_folds):
    # get the train and test set indicies for a given length of the dataset, seed and fold out of a given number of folds

    # exception: treat a single fold as having no test images
    if n_folds == 1:
        np.random.seed(seed)
        return np.random.permutation(n), np.arange(0)

    # set up number of test samples per fold, should cover entire data and not
    # differ by more than one
    min_n_test = n // n_folds
    remainder = n - min_n_test * n_folds
    nbs_test = np.ones(n_folds).astype(int) * min_n_test
    nbs_test[0:remainder] += 1

    assert nbs_test.sum() == n

    # choose a permutation
    np.random.seed(seed)
    perm = np.random.permutation(n)

    # compute start and end indices of each fold
    assert fold < n_folds, f"fold was {fold} but must not exceed {n_folds}."
    starts = np.concatenate([np.zeros(1).astype(int), np.cumsum(nbs_test)])


    # get data for the required fold
    start = starts[fold]
    end = starts[fold+1]

    train_idx = np.append(perm[:start], perm[end:])
    test_idx = perm[start:end]
    return train_idx, test_idx


def get_train_test_split(imgs, segs, file_names, seed=0, fold=0, n_folds=10):
    train_idx, test_idx = train_test_idx(len(imgs), seed, fold, n_folds)

    train_imgs = imgs[train_idx]
    train_labels = segs[train_idx]
    train_file_names = file_names[train_idx]

    test_imgs = imgs[test_idx]
    test_labels = segs[test_idx]
    test_file_names = file_names[test_idx]

    return train_imgs, train_labels, train_file_names, test_imgs, test_labels, test_file_names


# wrapper for cellpose training
def train_cellpose(imgs, segs, save_path, model_name, save_each=False, min_train_masks=0, batch_size = 32):
    model = models.CellposeModel(gpu=True,
                                 model_type="cyto2")
    channels = [0, 0]

    model.train(train_data=imgs.copy(),
                train_labels=segs.copy(),
                channels=channels,
                save_path=save_path,
                save_each=save_each,
                model_name=model_name,
                min_train_masks=min_train_masks,
                batch_size=batch_size)


def get_metrics(masks_true, masks_pred, thresholds=np.arange(11)/10):
    # compute metrics for all masks in a list of masks (e.g. different frames in a video)
    # thresholds are intersection over union thresholds

    aps, tps, fps, fns = [], [], [], []
    ious = []

    for ind in range(len(masks_true)):
        iout, pred = metrics.mask_ious(masks_true[ind].flatten(), masks_pred[ind].flatten())
        ious.extend(list(iout))

        ap, tp, fp, fn = metrics.average_precision(masks_true[ind].flatten(),
                                                   masks_pred[ind].flatten(),
                                                   threshold=thresholds)
        aps.append(ap)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
    aps = np.stack(aps)
    tps = np.stack(tps)
    fps = np.stack(fps)
    fns = np.stack(fns)

    ious = np.stack(ious)

    # update the ious by false positives
    ious = np.append(ious, np.zeros(int(fps[:, 0].sum())))
    return aps, tps, fps, fns, ious

def fold_from_name(metric_file_name):
    fn_list = metric_file_name.split("_")
    ind = fn_list.index("fold") + 1
    return int(fn_list[ind])

def mean_aps(a):
    # expects an array with axis n_imgs * [tps, fps, fns] * thresholds

    #a = a.mean(0)
    #return a[0] / (a[0] + a[1] + a[2])

    aps = a[:, 0] / (a[:, 0]+ a[:, 1]+ a[:, 2])
    return aps.mean(0)

def run_cellpose(imgs,
                 diameter,
                 cellprob_threshold=None,
                 flow_threshold=None,
                 mode="2D"):
    model = models.Cellpose(model_type='cyto2', gpu=True)
    channels = [[0,0]]
    if mode == "2D":
        imgs_flat = imgs.reshape(-1, *imgs.shape[-2:])
        masks, _, _, _ = model.eval([img for img in imgs_flat],
                            diameter=diameter,
                            flow_threshold=flow_threshold,
                            cellprob_threshold=cellprob_threshold,
                            channels=[0,0],
                            resample=True)

        return masks
    elif mode == "3D":
        masks, _, _, _ = model.eval([vol for vol in imgs],
                            diameter=diameter,
                            flow_threshold=flow_threshold,
                            cellprob_threshold=cellprob_threshold,
                            channels=[0,0],
                            resample=True,
                            do_3D=True,
                            anisotropy=3.76,
                            z_axis=0,
                            min_size=100000)
        masks = np.stack(masks)
        masks_flat = masks.reshape(-1, *masks.shape[-2:])
        return masks_flat

    elif mode == "stitch":
        masks, _, _, _ = model.eval([vol for vol in imgs],
                                     diameter=diameter,
                                     flow_threshold=flow_threshold,
                                     cellprob_threshold=cellprob_threshold,
                                     batch_size=16,
                                     channels=[0,0],
                                     resample=True,
                                     do_3D=False,
                                     stitch_threshold=0.1)
        masks = np.stack(masks)
        masks_flat = masks.reshape(-1, *masks.shape[-2:])
        return masks_flat

    print(f"Mode {mode} not recognized. Must be '2D', '3D' or 'stitch'")
