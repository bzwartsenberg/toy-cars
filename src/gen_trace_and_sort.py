"""Asdf."""
from car_model_single import CarModelSingle
from car_model_double import CarModelDouble
from car_model_n import CarModelN
from car_model_all_n import CarModelNAll
from train import EmbeddingConv2DStridedSimple
from types import SimpleNamespace
import pyprob
import torch

models = {
    'car_model_single': CarModelSingle,
    'car_model_double': CarModelDouble,
    'car_model_n': CarModelN,
    'car_model_n_all': CarModelNAll,
}


def trace_sort(trace):
    """Sort trace."""
    seqs = {1: [[], []]}

    # first pass to grab values
    for var in trace.variables:
        if 'x_sample' in var.name:
            n, at = [int(x) for x in var.name.split('_')[2:]]
            if at not in seqs.keys():
                seqs[at] = [[], []]
            seqs[at][0].append(var.value)
        if 'y_sample' in var.name:
            n, at = [int(x) for x in var.name.split('_')[2:]]
            seqs[at][1].append(var.value)

    # sort:
    rads = {k: torch.tensor(v).square().sum(dim=0) for k, v in seqs.items()}
    idx = {k: v.argsort() for k, v in rads.items()}

    # second pass to put values back:
    for var in trace.variables:
        if 'x_sample' in var.name:
            n, at = [int(x) for x in var.name.split('_')[2:]]
            var.value = seqs[at][0][idx[at][n].item()]
        if 'y_sample' in var.name:
            n, at = [int(x) for x in var.name.split('_')[2:]]
            var.value = seqs[at][1][idx[at][n].item()]
    return trace


def model_trace_generator_wrapper(original, modifier):
    """Wrap the trace generator in a modifier."""
    def func(*args, **kwargs):
        trace = next(original(*args, **kwargs))

        trace = modifier(trace)

        yield trace

    return func


def save_traces(args):
    """Save traces."""
    model = models[args.model_name](args)
    model._trace_generator = model_trace_generator_wrapper(
        model._trace_generator, trace_sort)

    model.save_dataset(args.dataset_dir,
                       num_traces=args.batch_size*args.num_batches,
                       num_traces_per_file=args.batch_size,
                       pickled=True,
                       compression='gzip')

    args.model = model


def train(args):
    """Train."""
    model = models[args.model_name](args)
    if args.sort_trace:
        model._trace_generator = model_trace_generator_wrapper(
            model._trace_generator, trace_sort)

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
        save_file_name_prefix=('output/inference_' + args.model_name),
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

    args.model_name = 'car_model_n_all'

    args.num_batches = 1000
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

    args.n_cars = 5  # note: this is only used for car_model_n
    args.spread = 5.
    args.lik_sigma = 0.1
    args.max_attempts = 30

    args.sort_trace = False

    train(args)
