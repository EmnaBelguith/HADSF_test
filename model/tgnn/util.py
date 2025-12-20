import csv
import json
from collections import OrderedDict

import torch
import numpy as np
import logging
import sys
import os
from datetime import datetime
from pytz import timezone, utc
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
import shutil, re


def custom_time(*args):
    utc_dt = utc.localize(datetime.utcnow())
    my_tz = timezone("Asia/Shanghai")
    converted = utc_dt.astimezone(my_tz)
    return converted.timetuple()


def change_optimizer_device(optimizer, device, dtype):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device).type(dtype)


def change_triplet_data_type(triplet):
    user = triplet[:, 0].astype(np.int64)
    item = triplet[:, 1].astype(np.int64)
    rating = triplet[:, 2].astype(np.float32)

    return [user, item, rating]


def copy_py_files(source_dir: str, dest_dir: str):
    py_files = [f.path for f in os.scandir(source_dir) if f.is_file() and f.path.endswith('.py')]
    for f in py_files:
        shutil.copy(f, dest_dir)


def change_tensor_device(device, *tensors):
    result = [x.to(device) for x in tensors]
    return result


def np_to_pt_tensor(device, *arrays):
    return [torch.from_numpy(x).to(device) for x in arrays]


def create_dirs(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def make_trainging_log_dir(base_dir: str, data_name: str, model_name: str):
    """
    create folders for training logging
    log/
    └── data_name
    	└── model_name
    	    ├── 2022-03-23-No0
    	    │   └── code
    	    └── 2022-03-23-No1
    		    └── code
    base_dir: 
    return base_dir/data_name/model_name/2022-12-12-No1, code_dir
    """
    dir_path = f"{base_dir}/{data_name}/{model_name}/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    exist_ids = [f.path.split('/')[-1] for f in os.scandir(dir_path) if f.is_dir()]
    max_id = 0

    if len(exist_ids) > 0:
        exist_ids = [re.search(r'(?<=No)[0-9]+', f) for f in exist_ids]
        exist_ids = [int(x.group(0)) for x in exist_ids if x is not None]
        max_id = max(exist_ids) + 1
    save_dir = f"{dir_path}/{datetime.now().strftime('%Y-%m-%d')}-No{max_id}/"
    code_save_dir = f"{save_dir}/code/"
    os.makedirs(save_dir)
    os.makedirs(code_save_dir)
    os.makedirs(f"{save_dir}/tb")
    return save_dir, code_save_dir


def get_logger(name, log_file_path=None, mode='w'):

    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt="%H:%M:%S")
    formatter.converter = custom_time
#     logging.Formatter.converter = customTime
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # FileHandler
    if log_file_path:
        dir_path = os.path.dirname(log_file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        file_handler = logging.FileHandler(log_file_path, mode=mode)
        file_handler.setLevel(level=logging.INFO)
    #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_tensorboard_writer(log_dir, model='w'):
    # if model == 'w' and os.path.exists(log_dir):
    #     shutil.rmtree(log_dir, ignore_errors=True)

    writer = SummaryWriter(log_dir, filename_suffix='tb')
    return writer


def get_args_str(args):
    attr = getattr(args, '__dict__')
    attr = dict(attr)
    # attr = {k: v for k, v in attr.items() if not k.startswith('_')}
    attr = {k: v for k, v in attr.items() if (type(v) in set([int, str, float])) and (not k.startswith('_'))}
    attr = json.dumps(attr, indent=4)
    return attr


def args_to_str(args):
    return get_args_str(args)


def args_to_dict(args):
    attr = getattr(args, '__dict__')
    attr = dict(attr)
    attr = {k: v for k, v in attr.items() if (type(v) in set([int, str, float])) and (not k.startswith('_'))}
    return attr


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or callable function

    Returns
    -------
    ret: callable function
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError
    else:
        return act


def get_optimizer(opt):
    if opt == 'SGD':
        return optim.SGD
    elif opt == 'Adam':
        return optim.Adam
    elif opt == 'AdamW':
        return optim.AdamW
    else:
        raise NotImplementedError


def to_etype_name(rating):
    return str(rating).replace('.', '_')


class MetricLogger(object):
    def __init__(self, attr_names, parse_formats, save_path):
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        self._file = open(save_path, 'w')
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow([parse_format % kwargs[attr_name]
                            for attr_name, parse_format in self._attr_format_dict.items()])
        self._file.flush()

    def close(self):
        self._file.close()


def torch_total_param_num(net):
    return sum([np.prod(p.shape) for p in net.parameters()])


def torch_net_info(net, save_path=None):
    info_str = 'Total Param Number: {}\n'.format(torch_total_param_num(net)) +\
               'Params:\n'
    for k, v in net.named_parameters():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str


def coo_matrix_to_sparse_tensor(m):
    values = m.data
    indices = np.vstack((m.row, m.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = m.shape

    return torch.sparse_coo_tensor(i, v, torch.Size(shape), dtype=torch.float32)
