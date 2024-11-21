import h5py
import os
import cellpose.models
import numpy as np
from utils import get_data, datasets_from_model, load_data, get_path
import ray

def run_cellpose(data_dir,
                 model_name,
                 model_dir=None,
                 datasets=None,
                 epoch=499,
                 stitch_threshold=0.1,
                 anisotropy=3.2,
                 chunk=0,
                 n_chunks=1):
    # segment a given chunk of the data with a trained model

    print(f"Running trained model {model_name}")

    # load trained model
    if model_dir is None:
        model_dir = data_dir

    model_name = model_name + f"_epoch_{epoch}"

    model = cellpose.models.CellposeModel(gpu=True,
                                          pretrained_model=os.path.join(model_dir,
                                                                        "models",
                                                                        model_name))

    # eval on datasets used for training if datasets are not specified
    if datasets is None:
        datasets = datasets_from_model(model_name)

    for dataset in datasets:
        imgs, segs, file_names = get_data(dataset,
                                          datasets,
                                          data_dir,
                                          chunk=chunk,
                                          n_chunks=n_chunks)

        # flatten all but the last three dimensions
        imgs_shape = imgs.shape
        imgs = imgs.reshape(-1, *imgs_shape[-3:])

        # predict according to train mode and save
        if "2D" in model_name:
            pred_masks, _, _ = model.eval(list(imgs.copy()),
                                          channels=[0, 0],
                                          batch_size=64,
                                          resample=True,
                                          augment=True,
                                          stitch_threshold=stitch_threshold)
        elif "3D" in model_name:
            aniso = anisotropy if "3D_iso" in model_name else None
            pred_masks, _, _ = model.eval(list(imgs.copy()),
                                          batch_size=32,
                                          channels=[0, 0],
                                          resample=True,
                                          augment=True,
                                          do_3D=True,
                                          anisotropy=aniso
                                          )

        # reshape back
        pred_masks = np.stack(pred_masks).reshape(*imgs_shape)

        # save
        for i in range(len(pred_masks)):
            eval_str = model_name.removeprefix("cellpose_")
            if "2D" in model_name:
                eval_str += f"_stitch_{stitch_threshold}"
            if "3D_iso" in model_name:
                eval_str += f"_aniso_{anisotropy}"

            res_dir = os.path.join(data_dir, dataset, "results", eval_str)

            if not os.path.exists(res_dir):
                os.mkdir(res_dir)
            with h5py.File(os.path.join(res_dir,
                                        f"{file_names[i]}_{eval_str}.h5"),
                           "w") as file:
                file.create_dataset("seg", data=pred_masks[i])
    print(f"Done running trained model {model_name}")

    return model_name


def main():
    model_dir = os.path.join(get_path(), "RBC_labelled/"  # change to the desired dataset
    )

    data_dir = os.path.join(get_path(), "RBC_labelled")  # change to the desired dataset

    datasets = ["wholelife"]
    model_names = [
        "cellpose_data_valid1_valid2_valid3_by_stack_mode_3D_iso_min_train_masks_0_seed_0_fold_0_of_1"  # change to the trained model's name
        # "cellpose_data_mask-ts_by_stack_mode_3D_iso_min_train_masks_0_seed_0_fold_0_of_1"
        # "cellpose_data_mask-r_mask-ts_by_stack_mode_3D_iso_min_train_masks_0_seed_0_fold_0_of_1"
    ]

    #TODO: change paths and model names to work on the example data

    stitch_threshold = 0.1
    anisotropy = 3.2
    epoch = 499
    n_gpus = 1
    n_chunks = n_gpus

    # load all required datasets
    shared_datasets = {}
    for dataset in datasets:
        imgs, segs, file_names = load_data(data_dir, dataset)
        shared_datasets[dataset] = {"imgs": imgs,
                                    "segs": segs,
                                    "file_names": file_names}

    # ray manages multiple gpus
    ray.init(include_dashboard=False,
             _redis_password="my_ray_password",  # change
             _temp_dir=os.path.expanduser("~/scratch/ray_temp"),
             num_gpus=n_gpus,
             num_cpus=40)

    run_cellpose_rmt = (ray.remote(num_gpus=1.0,
                                   num_cpus=2,
                                   max_calls=1))(run_cellpose)

    ready, unready = ray.wait([run_cellpose_rmt.remote(
                     data_dir=data_dir,
                     model_dir=model_dir,
                     datasets={dataset: shared_datasets[dataset] for dataset in datasets},
                     model_name=model_name,
                     stitch_threshold=stitch_threshold,
                     epoch=epoch,
                     anisotropy=anisotropy,
                     chunk=chunk,
                     n_chunks=n_chunks)
        for chunk in range(n_chunks)
        for model_name in model_names
    ], num_returns=1)

    # boilerplate for error handling
    while unready:
        try:
            ray.get(ready)
        except ray.exceptions.RayTaskError as e:
            print(f"{e.pid} crashed")
        ready, unready = ray.wait(unready, num_returns=1)

    ray.shutdown()


if __name__ == "__main__":
    main()
