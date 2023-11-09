import torch
import torch.utils.data
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_and_transform_data
from lib.normalization import (
    NScaler,
    MinMax01Scaler,
    MinMax11Scaler,
    StandardScaler,
    ColumnMinMaxScaler,
)
import numpy as np


def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == "max01":
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print("Normalize the dataset by MinMax01 Normalization")
    elif normalizer == "max11":
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print("Normalize the dataset by MinMax11 Normalization")
    elif normalizer == "std":
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print("Normalize the dataset by Standard Normalization")
    elif normalizer == "None":
        scaler = NScaler()
        data = scaler.transform(data)
        print("Does not normalize the dataset")
    elif normalizer == "cmax":
        # column min max, to be depressed
        # note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print("Normalize the dataset by Column Min-Max Normalization")
    else:
        raise ValueError
    return data, scaler


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio) :]
    val_data = data[
        -int(data_len * (test_ratio + val_ratio)) : -int(data_len * test_ratio)
    ]
    train_data = data[: -int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data


def data_loader(X, Y, batch_size, shuffle=False, drop_last=False):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    return dataloader


def get_dataloader(args, normalizer="std", single=True):
    data = load_and_transform_data(args)  # B, N, D

    data_train, data_val, data_test = split_data_by_ratio(
        data, args.val_ratio, args.test_ratio
    )

    # Save the arrays
    if args.save_arrays_EDA:
        np.save("data/processed/DSTGCRN/data_train.npy", data_train)
        np.save("data/processed/DSTGCRN/data_val.npy", data_val)
        np.save("data/DSTGCRN/data_test.npy", data_test)

    # normalize st data
    data_train[:, :, : args.normalizd_col], scaler = normalize_dataset(
        data_train[:, :, : args.normalizd_col], normalizer, args.column_wise
    )
    data_val[:, :, : args.normalizd_col], _ = normalize_dataset(
        data_val[:, :, : args.normalizd_col], normalizer, args.column_wise
    )
    data_test[:, :, : args.normalizd_col], _ = normalize_dataset(
        data_test[:, :, : args.normalizd_col], normalizer, args.column_wise
    )

    # add time window
    if args.horizon == 1:
        x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
        x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
        x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    else:
        x_tra, y_tra = Add_Window_Horizon(
            data_train, args.lag, args.horizon, single=False
        )
        x_val, y_val = Add_Window_Horizon(
            data_val, args.lag, args.horizon, single=False
        )
        x_test, y_test = Add_Window_Horizon(
            data_test, args.lag, args.horizon, single=False
        )

    print("Train: ", x_tra.shape, y_tra.shape)
    print("Val: ", x_val.shape, y_val.shape)
    print("Test: ", x_test.shape, y_test.shape)

    if not args.TNE:
        train_dataloader = data_loader(
            x_tra, y_tra, args.batch_size, shuffle=False, drop_last=True
        )
        val_dataloader = data_loader(
            x_val, y_val, args.batch_size, shuffle=False, drop_last=False
        )
        test_dataloader = data_loader(
            x_test, y_test, args.batch_size, shuffle=False, drop_last=False
        )
    else:
        train_dataloader = data_loader(
            x_tra, y_tra, args.batch_size, shuffle=False, drop_last=True
        )
        val_dataloader = data_loader(
            x_val, y_val, args.batch_size, shuffle=False, drop_last=True
        )
        test_dataloader = data_loader(
            x_test, y_test, args.batch_size, shuffle=False, drop_last=True
        )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        scaler,
    )
