import os
import numpy as np
import h5py
from utils.utils import get_metrics, datasets_from_model, get_data, get_eval_string, get_path, get_shells
import pickle


def evaluate_seg(data_dir,
                 datasets,
                 model_name,
                 stitch_threshold=0.1,
                 anisotropy=3.2,
                 shells=False):
    # evaluates an exisiting segmentation against the ground truth
    print(f"Evaluating segmentations of model {model_name}")

    # if no datasets are given, use those the model was trained on
    if datasets is None:
        datasets = datasets_from_model(model_name)

    for dataset in datasets:
        _, gt_segs, file_names = get_data(dataset, datasets, data_dir)

        metrics = np.zeros((len(file_names), 4, 11))

        eval_str = get_eval_string(model_name, stitch_threshold, anisotropy)

        for i, file_name in enumerate(file_names):
            with h5py.File(os.path.join(data_dir,
                                        dataset,
                                        "results",
                                        eval_str,
                                        f"{file_name}_{eval_str}.h5"),
                           "r") as file:
                pred_seg = file["seg"][:]
            # compute the metrics
            if not shells:
                if gt.max() == 0 and pred.max() == 0:
                    continue  # no segments, so aps ill-defined and tp, fp, fn count not affected
                aps, tps, fps, fns, _ = get_metrics([gt_segs[i]],
                                                        [pred_seg])
            else:
                # compute the shells first
                dilation_radius = 1
                erosion_radius = 1
                gt_shells = get_shells(gt_segs[i], erosion_radius, dilation_radius)
                pred_shells = get_shells(pred_seg, erosion_radius, dilation_radius)
                if gt_shells.max() == 0 and pred_shells.max() == 0:
                    continue  # no segments, so aps ill-defined and tp, fp, fn count not affected
                aps, tps, fps, fns, _ = get_metrics([gt_shells],
                                                    [pred_shells])

            metrics[i, :] = np.concatenate([aps, tps, fps, fns])

        d = {"metrics": metrics,
             "file_names": file_names}

        # save computed metrics
        filename_ending = "_shells" if shells else ""
        with open(os.path.join(data_dir,
                               dataset,
                               "results",
                               eval_str,
                               f"metrics_{eval_str}{filename_ending}.pkl"),
                  'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Done evaluating segmentations of model {model_name}")

    return 0


def main():
    seg_target  = "RBC"  # change to "parasite" for parasite data
    data_dir = os.path.join(get_path(), f"{seg_target}_labelled/")
    datasets = ["valid1", "valid2", "valid3"] #["mask-r", "mask-ts"]

    shells = False # set to True if you want to evaluate the shells instead of the full masks

    # give model names here
    model_names = [
        "cellpose_data_valid1_valid2_valid3_by_stack_mode_3D_iso_min_train_masks_0_seed_0_fold_0_of_10_epoch_499",
        "cellpose_data_valid1_valid2_valid3_by_stack_mode_3D_iso_min_train_masks_0_seed_0_fold_1_of_10_epoch_499",
        "cellpose_data_valid1_valid2_valid3_by_stack_mode_3D_iso_min_train_masks_0_seed_0_fold_2_of_10_epoch_499",
        "cellpose_data_valid1_valid2_valid3_by_stack_mode_3D_iso_min_train_masks_0_seed_0_fold_3_of_10_epoch_499",
        "cellpose_data_valid1_valid2_valid3_by_stack_mode_3D_iso_min_train_masks_0_seed_0_fold_4_of_10_epoch_499",
        "cellpose_data_valid1_valid2_valid3_by_stack_mode_3D_iso_min_train_masks_0_seed_0_fold_5_of_10_epoch_499",
        "cellpose_data_valid1_valid2_valid3_by_stack_mode_3D_iso_min_train_masks_0_seed_0_fold_6_of_10_epoch_499",
        "cellpose_data_valid1_valid2_valid3_by_stack_mode_3D_iso_min_train_masks_0_seed_0_fold_7_of_10_epoch_499",
        "cellpose_data_valid1_valid2_valid3_by_stack_mode_3D_iso_min_train_masks_0_seed_0_fold_8_of_10_epoch_499",
        "cellpose_data_valid1_valid2_valid3_by_stack_mode_3D_iso_min_train_masks_0_seed_0_fold_9_of_10_epoch_499",
    ]

    stitch_threshold = 0.1
    anisotropy = 3.2
    for model_name in model_names:
        evaluate_seg(data_dir=data_dir,
                     datasets=datasets,
                     model_name=model_name,
                     stitch_threshold=stitch_threshold,
                     anisotropy=anisotropy,
                     shells=shells)

if __name__ == "__main__":
    main()
