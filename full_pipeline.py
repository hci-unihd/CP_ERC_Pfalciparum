from utils import load_data, get_data, train_cellpose, slice_by_mode, train_test_idx, get_path
import numpy as np
import os
import ray
from train_cellpose import load_and_train
from evaluate import evaluate_seg
from run_cellpose import run_cellpose


def pipeline(data_path,
             datasets,
             mode,
             fold,
             n_folds=10,
             seed=0,
             anisotropy=3.2,
             min_train_masks=0,
             stitch_threshold=0.1,
             eval_epoch=499):
    # implements a full pipeline of training models with cross validation, segmenting and evaluating all data

    # train cellpose, model_name without epoch
    model_name = load_and_train(
        data_path=data_path,
        datasets=datasets,
        seed=seed,
        fold=fold,
        n_folds=n_folds,
        mode=mode,
        min_train_masks=min_train_masks,
        anisotropy=anisotropy)

    # run cellpose model to segment all data, model_name includes epoch
    model_name = run_cellpose(data_dir=data_path,
                              datasets=datasets,
                              model_name=model_name,
                              stitch_threshold=stitch_threshold,
                              epoch=eval_epoch)

    # evaluate the segmented data
    evaluate_seg(data_dir=data_path,
                 datasets=datasets,
                 model_name=model_name,
                 stitch_threshold=stitch_threshold,
                 anisotropy=anisotropy)

    return model_name


def main():
    #############################################
    # change parameters here as needed
    data_path = os.path.join(get_path(), "RBC_labelled")  # change data path as needed

    # list of lists of datasets on which the model is to be trained jointly
    datasets_list = [["valid1"], ["valid2"], ["valid3"],
                     ["valid1", "valid2", "valid3"]]  # ["mask-r", "mask-ts"]

    min_train_masks = 0
    seed = 0
    n_folds = 1
    mode = "3D_iso"  # 2D, 3D, 3D_iso
    anisotropy = 3.2
    stitch_threshold = 0.1
    eval_epoch = 499

    # ray manages the use of multiple GPUs for efficient training
    ray.init(include_dashboard=False,
             _redis_password="my_ray_password",  # change
             _temp_dir=os.path.expanduser("~/scratch/ray_temp"),
             num_gpus=10,  # set to the total number of GPUs available for training
             num_cpus=40)  # set to the total number of CPUs available for training

    # no parameters below should need changing
    #############################################################

    # load all required datasets
    unique_datasets = set(
        [item for sublist in datasets_list for item in sublist])

    shared_datasets = {}
    for dataset in unique_datasets:
        imgs, segs, file_names = load_data(data_path, dataset)
        shared_datasets[dataset] = {"imgs": imgs,
                                    "segs": segs,
                                    "file_names": file_names}

    pipeline_rmt = (ray.remote(num_gpus=1.0,
                               num_cpus=2,
                               max_calls=1))(pipeline)

    ready, unready = ray.wait([pipeline_rmt.remote(
        data_path=data_path,
        # create a dict containing exactly those datasets required for this job
        datasets={dataset: shared_datasets[dataset] for dataset in datasets},
        seed=seed,
        fold=fold,
        n_folds=n_folds,
        mode=mode,
        min_train_masks=min_train_masks,
        anisotropy=anisotropy,
        stitch_threshold=stitch_threshold,
        eval_epoch=eval_epoch
    )
        for fold in range(n_folds)
        for datasets in datasets_list
    ], num_returns=1)

    while unready:
        try:
            ray.get(ready)
        except ray.exceptions.RayTaskError as e:
            print(f"{e.pid} crashed")
        ready, unready = ray.wait(unready, num_returns=1)
    ray.shutdown()


if __name__ == "__main__":
    main()
