#!/usr/bin/env python3
from car_model_single import CarModel
from types import SimpleNamespace
import pyprob
from pyprob import util
from torch import nn
import torch
import numpy as np

def init_kaiming_normal(convlayer): #both for conv and linear
    torch.nn.init.kaiming_normal_(convlayer.weight, a=np.sqrt(5))
    if convlayer.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(convlayer.weight)
        bound = 1 / np.sqrt(fan_in)
        torch.nn.init.normal_(convlayer.bias, std=bound)

class EmbeddingConv2DStridedSimple(nn.Module):

    """Sort of the inverse of StyleGan
    """
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self._input_shape = util.to_size(input_shape)  # expecting 3d: [channels, height, width]
        self._output_shape = util.to_size(output_shape)
        input_channels = self._input_shape[0]
        self._output_dim = util.prod(self._output_shape)
        self._conv1 = nn.Conv2d(input_channels, 64, 3, stride=2)
        self._conv2 = nn.Conv2d(64, 64, 3, stride=2)
        self._conv3 = nn.Conv2d(64, 128, 3, stride=2)
        self._conv4 = nn.Conv2d(128, 128, 3, stride=2)
        self._conv5 = nn.Conv2d(128, 128, 3, stride=2)
        init_kaiming_normal(self._conv1)
        init_kaiming_normal(self._conv2)
        init_kaiming_normal(self._conv3)
        init_kaiming_normal(self._conv4)
        init_kaiming_normal(self._conv5)



        cnn_output_dim = self._forward_cnn(torch.zeros(self._input_shape).unsqueeze(0)).nelement()
        self._lin1 = nn.Linear(cnn_output_dim, self._output_dim)
        self._lin2 = nn.Linear(self._output_dim, self._output_dim)
        init_kaiming_normal(self._lin1)
        init_kaiming_normal(self._lin2)

    def _forward_cnn(self, x):
        x = torch.relu(self._conv1(x))
        x = torch.relu(self._conv2(x))
        x = torch.relu(self._conv3(x))
        x = torch.relu(self._conv4(x))
        x = torch.relu(self._conv5(x))
        return x

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(torch.Size([batch_size]) + self._input_shape)
        x = self._forward_cnn(x)
        x = x.view(batch_size, -1)
        x = torch.relu(self._lin1(x))
        x = torch.relu(self._lin2(x))
        return x.view(torch.Size([-1]) + self._output_shape)



def train(args):

    model = CarModel(args)

    model.learn_inference_network(
        num_traces=args.batch_size*args.num_batches,
        observe_embeddings={'depth_image_sample': {
            'dim': args.inf_em_hidden_dim,
            'reshape': [1, args.height, args.width],
            'embedding': 'custom',
            'custom_embedding': EmbeddingConv2DStridedSimple}},
        inference_network=pyprob.InferenceNetwork.LSTM,

        batch_size=args.batch_size,
        dataset_dir=args.dataset_dir,
        dataset_valid_dir=args.dataset_valid_dir,
        save_file_name_prefix='output/inference',
        proposal_mixture_components=args.inf_mixture_components,

        lstm_dim=args.inf_lstm_dim,
        lstm_depth=args.inf_lstm_depth,

        address_embedding_dim=args.inf_addr_em_dim,
        sample_embedding_dim=args.inf_samp_em_dim,
        distribution_type_embedding_dim=args.inf_dist_em_dim,

        learning_rate_init=args.inf_lr_init,

                                 )

    args.model = model


if __name__ == '__main__':

    args = SimpleNamespace()

    args.num_batches = 150000
    args.batch_size = 64
    args.dataset_dir = None
    args.dataset_valid_dir = None
    args.inf_mixture_components = 10
    args.inf_lstm_dim = 128
    args.inf_lstm_depth = 1
    args.inf_addr_em_dim = 24
    args.inf_lr_init = 1e-3

    args.inf_samp_em_dim = 24
    args.inf_dist_em_dim = 24

    args.inf_em_hidden_dim = 128
    args.width = 256
    args.height = 256

    train(args)
