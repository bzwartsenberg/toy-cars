from sacred import Experiment
from types import SimpleNamespace
from pprint import pprint
import wandb
import os
import sys
import time
import subprocess

from gen_trace_and_sort import train

import pyprob
import torch
import numpy as np
import random

# wandb setup:
ex = Experiment('toy-cars')
WANDB_PROJECT_NAME = 'toy-cars'
if '--unobserved' in sys.argv:
    os.environ['WANDB_MODE'] = 'dryrun'

@ex.config
def my_config():
    home_dir = './'
    artifact_dir = './output/'

    # paths
    model_dir = artifact_dir + 'models/' + pyprob.util.get_time_stamp() + '/'
    os.makedirs(model_dir, exist_ok=True)

    model_name = 'car_model_n_all'

    num_batches = 1000
    batch_size = 64
    dataset_dir = None
    dataset_valid_dir = None
    inf_mixture_components = 10
    inf_lstm_dim = 128
    inf_lstm_depth = 1
    inf_addr_em_dim = 24
    inf_lr_init = 1e-3

    inf_samp_em_dim = 24
    inf_dist_em_dim = 24

    inf_em_hidden_dim = 128
    width = 256
    height = 256

    n_cars = 5  # note: this is only used for car_model_n
    spread = 5.
    lik_sigma = 0.1
    max_attempts = 30

    sort_trace = False

def seed_all(seed):  # thanks Vaden
    """Seed all devices deterministically off of seed independently."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pyprob.set_random_seed(seed)


def init(config, run):
    """Init."""
    args = SimpleNamespace(**config)
    pprint(args.__dict__)
    args.run = run

    seed_all(args.seed)

    return args


def main(args):
    """Run main."""
    if args.train:
        train(args)


@ex.automain
def my_main(_config, _run):
    """Run automain."""
    args = init(_config, _run)
    wandb_run = wandb.init(project=WANDB_PROJECT_NAME,
                            config=_config,
                              tags=[_run.experiment_info['name']])
    args.wandb_run = wandb_run
    main(args)
