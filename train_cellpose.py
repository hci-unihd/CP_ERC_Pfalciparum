from utils import load_data, get_data, get_path, train_cellpose, slice_by_mode, train_test_idx
import numpy as np
import os
import ray


def load_and_split(data_path, datasets, fold=0, n_folds=10, seed=0):
    # loads data from multiple datasets and creates train / test split
    imgs = []
    segs = []
    file_names = []
    dataset_size_acc = 0
    train_idx = []
    test_idx = []
    for dataset in datasets:
        i, s, fn = get_data(dataset, datasets, data_path)
        imgs.append(i)
        segs.append(s)
        file_names.append(fn)

        # get train and test indices for this dataset
        train_idx_dataset, test_idx_dataset = train_test_idx(len(i),
                                                             seed=seed,
                                                             fold=fold,
                                                             n_folds=n_folds)
        train_idx_dataset += dataset_size_acc  # shift indices by size of previous datasets
        test_idx_dataset += dataset_size_acc

        train_idx.append(train_idx_dataset)
        test_idx.append(test_idx_dataset)
        dataset_size_acc += len(i)  # increase loaded dataset size

    # turn lists into np.array
    imgs = np.concatenate(imgs)
    segs = np.concatenate(segs)
    file_names = np.concatenate(file_names)
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)

    # split into train and test
    train_imgs = imgs[train_idx]
    train_segs = segs[train_idx]
    train_file_names = file_names[train_idx]

    test_imgs = imgs[test_idx]
    test_segs = segs[test_idx]
    test_file_names = file_names[test_idx]
    return train_imgs, train_segs, train_file_names, test_imgs, test_segs, test_file_names


def load_and_train(data_path,
                   datasets,
                   mode="3D_iso",
                   fold=0,
                   n_folds=10,
                   seed=0,
                   min_train_masks=0,
                   anisotropy=3.2):
    # loads the data and trains a model
    # set up model name
    if isinstance(datasets, list):
        datasets_str = "_".join(datasets)
    elif isinstance(datasets, dict):
        l = list(datasets.keys())
        l.sort()
        datasets_str = "_".join(l)
    model_name = f"cellpose_data_{datasets_str}_by_stack_mode_{mode}_min_train_masks_{min_train_masks}_seed_{seed}_fold_{fold}_of_{n_folds}"


    # check if model is already trained
    if os.path.isfile(os.path.join(data_path,
                                   "models",
                                   model_name + "_epoch_499")):
        print(f"{model_name} already done :)!")
        return model_name


    # load and split data
    train_imgs, train_segs, train_file_names, \
    test_imgs, test_segs, test_file_names = load_and_split(data_path,
                                                           datasets,
                                                           fold=fold,
                                                           n_folds=n_folds,
                                                           seed=seed)



    # slice the 3D stacks as needed for training with cellpose
    # Slices of the same stack will be next to each other, but that does not matter
    # as cellpose shuffles during training.
    train_imgs, train_segs, train_file_names = slice_by_mode(train_imgs,
                                                             train_segs,
                                                             train_file_names,
                                                             mode=mode,
                                                             anisotropy=anisotropy)

    # train model
    print(f"Training model {model_name}")

    train_cellpose(train_imgs,
                   train_segs,
                   save_path=data_path,
                   save_each=True,
                   model_name=model_name,
                   min_train_masks=min_train_masks)
    print(f"Done training model {model_name}")
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
    n_folds = 10
    mode = "3D_iso"  # 2D, 3D, 3D_iso
    anisotropy = 3.2

    # ray manages the use of multiple GPUs for efficient training
    ray.init(include_dashboard=False,
             _redis_password="my_ray_password",  # change
             _temp_dir=os.path.expanduser("~/scratch/ray_temp"),
             num_gpus=10,  # set to the total number of GPUs available for training
             num_cpus=40)  # set to the total number of CPUs available for training

    # no parameter below show need changing
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


    load_and_train_rmt = (ray.remote(num_gpus=1.0,
                                 num_cpus=2,
                                 max_calls=1))(load_and_train)

    ready, unready = ray.wait([load_and_train_rmt.remote(
        data_path=data_path,
        # create a dict containing exactly those datasets required for this job
        datasets={dataset: shared_datasets[dataset] for dataset in datasets},
        seed=seed,
        fold=fold,
        n_folds=n_folds,
        mode=mode,
        min_train_masks=min_train_masks,
        anisotropy=anisotropy)

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