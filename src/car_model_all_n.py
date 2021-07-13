#!/usr/bin/env python3
from car_model_base import BaseCarModel
import pyprob
import torch
from pyprob.distributions import Normal
from types import SimpleNamespace
import matplotlib.pyplot as plt
from util import Car


class CarModelNAll(BaseCarModel):
    """car model with single car."""

    def __init__(self, args):
        """asdf."""
        super().__init__(args)

    def forward(self, verbose=False, generate_samples=False):
        """Forward function."""

        number_of_vehicles = self.args.n_cars

        if verbose:
            print('\n\nNumber of other vehicles to spawn: ',
                  number_of_vehicles)

        attempts = 1

        while True:
            self.reset_model()
            # adding ego vehicle:
            self.accepted_cars.append(Car(0., 0.))

            if verbose:
                print('Attempt number: ', attempts)

            x_pos = torch.zeros(args.n_cars).to(pyprob.util._device)
            y_pos = torch.zeros(args.n_cars).to(pyprob.util._device)
            x_scale = torch.ones(args.n_cars).to(pyprob.util._device) \
                * self.args.spread
            y_scale = torch.ones(args.n_cars).to(pyprob.util._device) \
                * self.args.spread

            x_dist = Normal(x_pos, x_scale)
            y_dist = Normal(y_pos, y_scale)

            x = x_dist.sample()
            y = y_dist.sample()

            if self.args.sort:
                pass
                # TODO: sort here?

            success = True
            for i in range(int(number_of_vehicles)):
                if verbose:
                    print('Spawning car', i, 'out of',
                          int(number_of_vehicles))

                success &= self.attempt_to_place_car(x[i], y[i])

                if not success:
                    break  # stop trying

            attempts += 1
            if attempts > self.args.max_attempts or success:
                break

        image = torch.FloatTensor(self.to_image()).to(pyprob.util._device)

        scale = torch.tensor(self.args.lik_sigma).to(pyprob.util._device)
        likelihood = Normal(image, scale)
        pyprob.observe(likelihood, name="depth_image_sample",constants={'scale'
                                                                        :
                                                                        scale})
        if generate_samples:
            return number_of_vehicles, image
        else:
            return number_of_vehicles


if __name__ == '__main__':


    for i in range(10):
        pyprob.set_random_seed(i)

        args = SimpleNamespace()

        args.n_cars = 5
        args.sort = False
        args.lik_sigma = 0.1
        args.spread = 5.
        args.max_attempts = 100
        model = CarModelNAll(args)

        number_of_vehicles, image = model.get_trace(generate_samples=True, verbose=True).result

        fig,ax = plt.subplots(figsize=(10,10))

        print('Spawned number of vehicles: ', number_of_vehicles)

        ax.pcolormesh(model.x, model.y, image.detach().numpy(), shading='auto')
        plt.show()

        # plt.savefig('img/test.png')
