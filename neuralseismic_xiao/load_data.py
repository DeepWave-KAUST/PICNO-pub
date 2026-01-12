import torch
import numpy as np
from .utilis import find_files
from torch.utils.data import DataLoader, random_split, TensorDataset

def load_helmholtz_small(
    data_path,
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    test_resolutions=[70],
    train_resolution=70,
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=False,
    encoding="channel-wise",
    channel_dim=1,
    train_files_len = 5,
    test_files_len = 2,
    train_suffix = '*_constantv.pt',
    test_suffix = '*_test_constantv.pth',
):
    """Loads a small Helmhotlz dataset

    Training contains 1000 samples.
    Testing contains 100 samples.

    Parameters
    ----------
    n_train : int
    n_tests : int
    batch_size : int
    test_batch_sizes : int list
    test_resolutions : int list, default is [16, 32],
    grid_boundaries : int list, default is [[0,1],[0,1]],
    positional_encoding : bool, default is True
    encode_input : bool, default is False
    encode_output : bool, default is True
    encoding : 'channel-wise'
    channel_dim : int, default is 1
        where to put the channel dimension, defaults size is 1
        i.e: batch, channel, height, width

    Returns
    -------
    training_dataloader, testing_dataloaders

    training_dataloader : torch DataLoader
    testing_dataloaders : dict (key: DataLoader)
    """
    return load_helmholtz_pt(
        data_path=data_path,
        n_train=n_train,
        n_tests=n_tests,
        batch_size=batch_size,
        test_batch_size=test_batch_sizes,
        test_resolution=test_resolutions,
        train_resolution=train_resolution,
        grid_boundaries=grid_boundaries,
        positional_encoding=positional_encoding,
        encode_input=encode_input,
        encode_output=encode_output,
        encoding=encoding,
        channel_dim=channel_dim,
        test_files_len=test_files_len,
        train_files_len=train_files_len,
        train_suffix=train_suffix,
        test_suffix=test_suffix,
    )


def load_helmholtz_pt(
    data_path,
    n_train,
    n_tests,
    batch_size,
    test_batch_size,
    test_files_len = 2,
    train_files_len = 5,
    test_resolution=70,
    train_resolution=70,
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=False,
    encoding='channel-wise',
    channel_dim=1,
    train_suffix = '*_constantv.pt',
    test_suffix = '*_test_constantv.pth',
):
    # min_v, max_v = 1.500, 5.000
    train_files = find_files(data_path,train_suffix)
    test_files = find_files(data_path,test_suffix)
    input_key = ['U0_real','U0_imag', 'v_true', 'v_0', 'frequency']
    label_key = ['du_real','du_imag']
    x_train, y_train = [], []
    for i in train_files[:train_files_len]:
        print(f'loading files: {i}')
        train_data_all = torch.load(i)
        x_train.append(torch.cat([train_data_all[input_key[0]].unsqueeze(channel_dim).transpose(2,3), train_data_all[input_key[1]].unsqueeze(channel_dim).transpose(2,3), train_data_all[input_key[2]].unsqueeze(channel_dim),train_data_all[input_key[3]].unsqueeze(channel_dim), torch.tensor(train_data_all[input_key[4]]).reshape(-1,1).repeat(1,int(train_data_all[input_key[3]].shape[0]/len(train_data_all[input_key[4]]))).reshape(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(channel_dim).repeat(1,1,train_resolution,train_resolution)], dim=1))
        y_train.append(torch.cat([train_data_all[label_key[0]].unsqueeze(channel_dim).transpose(2,3), train_data_all[label_key[1]].unsqueeze(channel_dim).transpose(2,3)], dim=1)) 
    x_train = torch.cat(x_train, dim=0).to(torch.float32)
    y_train = torch.cat(y_train, dim=0).to(torch.float32)
    
    x_test, y_test = [], []
    for i in test_files[:test_files_len]:
        print(f'loading testing files: {i}')
        test_data_all = torch.load(i)
        x_test.append(torch.cat([test_data_all[input_key[0]].unsqueeze(channel_dim).transpose(2,3), test_data_all[input_key[1]].unsqueeze(channel_dim).transpose(2,3), test_data_all[input_key[2]].unsqueeze(channel_dim),test_data_all[input_key[3]].unsqueeze(channel_dim), torch.tensor(test_data_all[input_key[4]]).reshape(-1,1).repeat(1,int(train_data_all[input_key[3]].shape[0]/len(train_data_all[input_key[4]]))).reshape(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(channel_dim).repeat(1,1,test_resolution,test_resolution)], dim=1))
        y_test.append(torch.cat([test_data_all[label_key[0]].unsqueeze(channel_dim).transpose(2,3), test_data_all[label_key[1]].unsqueeze(channel_dim).transpose(2,3)], dim=1))
        
    x_test = torch.cat(x_test, dim=0).to(torch.float32)
    y_test = torch.cat(y_test, dim=0).to(torch.float32)
    
    train_indexs, test_indexs = np.random.permutation(len(x_train))[:n_train], np.random.permutation(len(x_test))[:n_tests]
    train_indexs, test_indexs = np.arange(len(x_train))[:n_train], np.arange(len(x_test))[:n_tests] # for the paper
    x_train, y_train = x_train[train_indexs], y_train[train_indexs]
    x_test, y_test = x_test[test_indexs], y_test[test_indexs]
    
    
    input_encoder = None
    output_encoder = None
        
    train_db = TensorDataset(
        x_train,
        y_train,
    )
    
    valid_size = int(0.1 * len(train_db))  # 例如，分割出10%作为验证集
    train_size = len(train_db) - valid_size

    # 随机分割数据集
    train_dataset, valid_dataset = random_split(train_db, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True, drop_last=True)
    
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=False, drop_last=True)
    
    if input_encoder is not None:
        x_test = input_encoder.encode(x_test)

    test_db = TensorDataset(
            x_test,
            y_test,
    )
    test_loader = DataLoader(
        test_db,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    
    return train_loader, val_loader, test_loader, output_encoder    

def normalize(v, max_v, min_v):
    return 2.0 * (v - min_v) / (max_v - min_v) - 1.0
        
    
    

