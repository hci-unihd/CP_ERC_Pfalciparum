import h5py
import os
import cellpose.models
from cellpose.utils import stitch3D
import numpy as np
import skimage

from utils import get_data, datasets_from_model, load_data, handle_small_segments_old, \
    smooth_segments, relabel, load_predictions, get_shells, handle_small_segments_simple,\
    merge_in_rbc, get_path


######################################################
# Make changes to the config dict in the post_process function or pass the parameters explicitly
# Make changes to the main function to load the correct data and models
######################################################



def post_process(seg, stage="rbc", rbc_seg=None, min_size=None, nbhd_size=None, radius=None, do_3D=None, stitch_threshold=0.1):
    #############################################
    # change parameters here as needed

    # configuration dicts for different post-processing options
    config = {"rbc": {"min_size": 700,
                      "nbhd_size": 1,
                      "radius": 1,
                      "do_3D": False},
              "rbc_old": {"min_size": 700,
                          "nbhd_size": 1,
                          "radius": 1,
                          "do_3D": False},

              "parasite": {"radius": 0,
                           "do_3D": True,
                           "nbhd_size": None,
                           "min_size": None}}

    # nothing else in this function should need changing
    #############################################


    assert stage == "rbc" or stage == "rbc_old" or rbc_seg is not None

    # load parameters from config dicts, if they are not explicitly given
    if min_size is None:
        min_size = config[stage]["min_size"]
    if nbhd_size is None:
        nbhd_size = config[stage]["nbhd_size"]
    if radius is None:
        radius = config[stage]["radius"]
    if do_3D is None:
        do_3D = config[stage]["do_3D"]

    # do the post-processing, first handling the filtering / merging, then the smoothing
    new_seg = np.zeros_like(seg)
    for t, frame in enumerate(seg):
        if stage == "rbc":
            tmp_seg = handle_small_segments_simple(frame, nbhd_size)
        elif stage == "rbc_old":
            tmp_seg = handle_small_segments_old(frame, min_size)
        elif stage == "parasite":
            tmp_seg = merge_in_rbc(frame, rbc_seg[t])
        else:
            raise NotImplementedError(f"Stage must be 'rbc', 'rbc_old' or 'parasite'")
        new_seg[t] = smooth_segments(tmp_seg, radius, do_3D)

    # stitch across time and relabel to continguous labels
    if stage == "rbc":
        new_seg = stitch3D(new_seg, stitch_threshold)
        new_seg = relabel(new_seg)  # only need to relabel for rbc, parasites only have label 1

    return new_seg


def main():

    #############################################
    # change parameters here as needed
    data_dir = os.path.join(get_path(), "data_20221025")
    datasets = ["wholelife"]
    post_process_name = "pp_rbc_shell_ring_joint"  # give a name for this postprocessing (used for the filenames of the result)

    # specify the names of the models used for rbc, late and ring stages
    exp_str_rbc = "data_valid1_valid2_valid3_by_stack_mode_3D_iso_min_train_masks_0_seed_0_fold_0_of_1_epoch_499_aniso_3.2"
    exp_str_parasite_late = "data_mask-ts_by_stack_mode_3D_iso_min_train_masks_0_seed_0_fold_0_of_1_epoch_499_aniso_3.2"
    exp_str_parasite_ring = "data_mask-r_mask-ts_by_stack_mode_3D_iso_min_train_masks_0_seed_0_fold_0_of_1_epoch_499_aniso_3.2"

    # nothing else in this function should need changing
    #############################################

    for dataset in datasets:
        # load data and predictions
        imgs, _, file_names = load_data(data_dir, dataset)
        pred_rbc = load_predictions(data_dir, dataset, file_names, exp_str_rbc)
        pred_para_ring = load_predictions(data_dir, dataset, file_names, exp_str_parasite_ring)
        pred_para_late = load_predictions(data_dir, dataset, file_names, exp_str_parasite_late)

        # do the postprocessing
        for i, (seg_rbc, seg_para_ring, seg_para_late) in enumerate(zip(pred_rbc, pred_para_ring, pred_para_late)):
            pp_pred_rbc = post_process(seg_rbc, stage="rbc")
            pp_pred_para_ring = post_process(seg_para_ring, stage="parasite", rbc_seg=pp_pred_rbc)
            pp_pred_para_late = post_process(seg_para_late, stage="parasite", rbc_seg=pp_pred_rbc)

            # compute shells
            shells_rbc = np.zeros_like(pp_pred_rbc)
            for t, frame in enumerate(pp_pred_rbc):
                shells_rbc[t] = get_shells(pp_pred_rbc[t])

            # stack everything and save as images
            stack = np.stack([imgs[i],
                              pp_pred_rbc,
                              shells_rbc,
                              pp_pred_para_ring,
                              pp_pred_para_late
                              ], axis=2)  # fiji needs (t)zcxy

            path_to_stack = os.path.join(data_dir,
                                         dataset,
                                         "results",
                                         post_process_name)

            if not os.path.exists(path_to_stack):
                os.mkdir(path_to_stack)

            skimage.io.imsave(
                os.path.join(path_to_stack, file_names[i] + "_preds.tiff"), stack,
                imagej=True)


if __name__ == "__main__":
    main()
