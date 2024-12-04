import h5py
import os
import cellpose.models
import numpy as np
from utils.utils import get_data, datasets_from_model, load_data, get_path
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

    if epoch is not None:
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
        else:
            aniso = anisotropy if ("3D_iso" in model_name or model_name.endswith("model")) else None
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

    model_path = os.path.join(get_path(), "..")  # change to the desired model directory
    data_dir = os.path.join(get_path(), "sample_data")  # change to the desired dataset

    datasets = ["sample_subset"]
    model_names = [
        "erythrocyte_model",  # change to the trained model's name, do not include the epoch
        "late_stage_model",
        "joint_model",
    ]

    stitch_threshold = 0.1
    anisotropy = 3.2
    epoch = None
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
             num_gpus=n_gpus,
             num_cpus=40)

    run_cellpose_rmt = (ray.remote(num_gpus=1.0,
                                   num_cpus=2,
                                   max_calls=1))(run_cellpose)

    ready, unready = ray.wait([run_cellpose_rmt.remote(
                     data_dir=data_dir,
                     datasets={dataset: shared_datasets[dataset] for dataset in datasets},
                     model_name=model_name,
                     model_dir=model_path,
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

    # save the results as tiff files, adapt directory and file names when running this script directly
    dataset = "sample_subset"
    import skimage.io
    import matplotlib.pyplot as plt

    raw_file = os.path.join(data_dir, dataset, "data", "sample_stack.h5")

    rbc_file = os.path.join(data_dir, dataset, "results",
                            "erythrocyte_model",
                            "sample_stack_erythrocyte_model.h5")

    para_late_file = os.path.join(data_dir, dataset, "results",
                                  "late_stage_model",
                                  "sample_stack_late_stage_model.h5")

    para_joint_file = os.path.join(data_dir, dataset, "results",
                                   "joint_model",
                                   "sample_stack_joint_model.h5")


    with h5py.File(raw_file, "r") as f:
        raw = f["data"][:][0]
    with h5py.File(rbc_file, "r") as f:
        rbc_seg = f["seg"][:][0]
    with h5py.File(para_late_file, "r") as f:
        para_late_seg = f["seg"][:][0]
    with h5py.File(para_joint_file, "r") as f:
        para_joint_seg = f["seg"][:][0]

    fig, ax = plt.subplots(1, 4, figsize=(10, 3))
    z_slice = 20

    file_name = "sample_stack"
    fig_path = os.path.join(get_path(), "figures")

    ax[0].imshow(raw[z_slice], cmap="gray")
    ax[0].set_title("Raw")
    ax[0].axis("off")
    ax[1].imshow(rbc_seg[z_slice], cmap="tab20", interpolation="none")
    ax[1].set_title("Erythrocyte Model")
    ax[1].axis("off")
    ax[2].imshow(para_late_seg[z_slice], cmap="tab20", interpolation="none")
    ax[2].set_title("Late Stage Model")
    ax[2].axis("off")
    ax[3].imshow(para_joint_seg[z_slice], cmap="tab20", interpolation="none")
    ax[3].set_title("Joint Model")
    ax[3].axis("off")
    fig.suptitle(f"Sample stack z-slice {z_slice}")
    fig.savefig(os.path.join(fig_path, file_name + f"_z_slice_{z_slice}_preds.png"))


    stack = np.stack([raw,
                      rbc_seg,
                      para_late_seg,
                      para_joint_seg])

    stack = np.moveaxis(stack, 0, 1)

    skimage.io.imsave(os.path.join(fig_path, file_name + "_preds.tiff"), stack, imagej=True)

if __name__ == "__main__":
    main()
