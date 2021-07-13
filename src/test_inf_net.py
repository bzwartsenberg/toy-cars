#!/usr/bin/env python3
from types import SimpleNamespace
import pyprob
from pyprob import util
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from train import models

from train import EmbeddingConv2DStridedSimple


def posns_from_trace(trace):
    """Generate the positions from trace"""
    posns = []

    for i in range((len(trace.variables)-1)//2):
        var_x = trace.variables[2*i]
        var_y = trace.variables[2*i+1]

        car_i = int(var_x.name.split('_')[2])

        xy = (var_x.value.item(), var_y.value.item())

        if len(posns) <= car_i:
            posns.append(xy)  # if it's first, append it
        else:
            posns[car_i] = xy  # else overwrite
    return posns


def get_posterior(args, observation):
    """generate the posterior based on an input."""
    model = models[args.model_name](args)

    model.load_inference_network(args.loadnetwork)

    posterior = model.posterior(
        num_traces=500,
        inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK,
        observe={'depth_image_sample': observation},
                               )

    return posterior

def gauss_kernel(x_tr, y_tr, XX, YY):
    """Gaussian kernel"""
    eps = 0.2
    Z = torch.exp(-((XX - x_tr)**2 + (YY - y_tr)**2)/(2*(eps**2)))
    return Z


def car_kernel(x_tr, y_tr, XX, YY):
    """car sized kernel"""
    Z = torch.zeros_like(XX)
    d = 1.
    cond = (XX > (x_tr - d/2)) & (XX < (x_tr + d/2))
    cond = cond & (YY > (y_tr - d/2)) & (YY < (y_tr + d/2))

    Z[cond] = 1.

    return Z

def get_box(x_tr, y_tr, d=1.):
    """make a box for plotting"""
    xs = np.array([-1., 1., 1., -1., -1.])
    ys = np.array([-1., -1., 1., 1., -1.])

    xs = xs*d/2 + x_tr
    ys = ys*d/2 + y_tr

    return xs, ys



def kde(posterior, kernel=car_kernel):
    """provide a KDE for where the car is"""
    x = np.linspace(-10, 10, 256)
    y = np.linspace(-10, 10, 256)

    XX, YY = np.meshgrid(x, y)

    XX = torch.tensor(XX)
    YY = torch.tensor(YY)

    def expectation_func(trace):
        Z = torch.zeros_like(XX)
        posns = posns_from_trace(trace)
        for xy in posns:
            x_tr = xy[0]
            y_tr = xy[1]

            Z += kernel(x_tr, y_tr, XX, YY)

        return Z

    kde_out = posterior.expectation(expectation_func)

    return x, y, kde_out


def test_single_car():
    """Do the testing."""
    args = SimpleNamespace()
    args.loadnetwork = 'output/inference_car_model_single_20210630_141630_traces_617152.network'
    args.seed = 42
    args.model_name = 'car_model_single'

    posns, x, y, image_obs = generate_observation(args)

    plt.imshow(image_obs)

    print('True posisions are:')
    for i, xy in enumerate(posns):
        print('True x{} is {}'.format(i, xy[0]))
        print('True y{} is {}'.format(i, xy[1]))

    posterior = get_posterior(args, image_obs)

    def xy_from_trace(trace):
        posns = posns_from_trace(trace)
        # note this only works for single cars obvs
        x, y = posns[0]

        return torch.tensor([x, y])

    xy_exp = posterior.expectation(xy_from_trace)

    print('posterior mean x is {}'.format(xy_exp[0].item()))
    print('posterior mean y is {}'.format(xy_exp[1].item()))


def generate_observation(args):
    """Create an observation using seed."""
    model = models[args.model_name](args)

    pyprob.set_random_seed(args.seed)

    trace = model.get_trace(generate_samples=True, verbose=True)

    number_of_vehicles, image_obs = trace.result

    posns = posns_from_trace(trace)

    return posns, model.x, model.y, image_obs


def test_with_kde(args):
    """Do the testing."""

    posns, x_obs, y_obs, image_obs = generate_observation(args)
    print(posns)
    print('True posisions are:')
    for i, xy in enumerate(posns):
        print('True x{} is {}'.format(i, xy[0]))
        print('True y{} is {}'.format(i, xy[1]))

    posterior = get_posterior(args, image_obs)

    print('Posterior effective sample size:', posterior.effective_sample_size.item())

    x_kde, y_kde, kde_im = kde(posterior)

    # plotting:
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].pcolormesh(x_obs, y_obs, image_obs, shading='auto')
    ax[0].set_title('observation')

    ax[1].pcolormesh(x_kde, y_kde, kde_im, shading='auto')
    ax[1].set_title('posterior KDE')

    for i, xy in enumerate(posns):
        xs, ys = get_box(xy[0], xy[1])
        ax[1].plot(xs, ys, lw=0.5, c='r')
    xs, ys = get_box(0.,0.)
    ax[1].plot(xs, ys, lw=0.5, c='g')
    ax[0].plot(xs, ys, lw=0.5, c='g')

    if args.savename is not None:
        plt.savefig('img/' + args.savename, dpi=300)

    plt.show()


if __name__ == '__main__':
    args = SimpleNamespace()

    # args.model_name = 'car_model_double'
    # args.model_name = 'car_model_single'
    args.model_name = 'car_model_n'
    args.seed = 42
    args.savename = 'double_test.png'
    args.n_cars = 4
    args.lik_sigma = 0.1

    # car_model_single 640K samples:
    # args.loadnetwork = 'output/inference_car_model_single_20210630_141630_traces_617152.network'
    # car_model_single 64K samples:
    # args.loadnetwork = 'output/inference_car_model_single_20210629_150909_traces_65600.network'
    # car_model_n (n=2) 64K samples:
    # args.loadnetwork = 'output/inference_car_model_n_20210630_153949_traces_64000.network'
    # car_model_n (n=4) 64K samples:
    args.loadnetwork = 'output/inference_car_model_n_20210630_165008_traces_64000.network'
