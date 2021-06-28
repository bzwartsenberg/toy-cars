#!/usr/bin/env python3

from car_model_base import BaseCarModel




class CarModel(BaseCarModel):
    """car model with single car"""

    def __init__(self, args):
        super().__init__(args)

        self.args.spread = 5.
        self.args.max_attempts = 100
        self.args.lik_sigma = 0.05

    def forward(self, verbose=False, generate_samples=False):
        """Forward function"""

        #note: as soon as I have a poisson surrogate/proposal up this should change
        poisson_center = torch.tensor(3.).to(pyprob.util._device)
        num_vehicles_dist = Normal(poisson_center,
                                   torch.tensor(3)).to(pyprob.util._device)
        max_vehicles = torch.tensor(14.)
        num_vehicles_float = pyprob.sample(num_vehicles_dist,
                                           name="num_vehicles_float_sample",
                                           constants=num_vehicles_dist.get_input_parameters())
        number_of_vehicles = torch.max(torch.tensor(0.),
                                       torch.min(torch.round(num_vehicles_float),
                                       max_vehicles))

        if verbose:
            print('\n\nNumber of other vehicles to spawn: ',
                  number_of_vehicles.item())

        x_dist = Normal(torch.tensor(0.).to(pyprob.util._device),
                         torch.tensor(self.args.spread).to(pyprob.util._device))
        y_dist = Normal(torch.tensor(0.).to(pyprob.util._device),
                         torch.tensor(self.args.spread).to(pyprob.util._device))

        # adding ego vehicle:
        self.accepted_cars.append(Car(0.,0.))

        for i in range(int(number_of_vehicles.item())):
            attempts = 0
            while True:
                if verbose:
                    print('Spawning car',i, 'out of',
                          int(number_of_vehicles.item()))
                    print('Attempt number: ', attempts)
                x = pyprob.sample(x_dist, name="x_sample_{}_{}".format(i, attempts),
                                constants=x_dist.get_input_parameters()).to(pyprob.util._device)
                y = pyprob.sample(y_dist, name="y_sample_{}_{}".format(i, attempts),
                                constants=y_dist.get_input_parameters()).to(pyprob.util._device)

                if not Car(x.item(),
                           y.item()).overlaps_with_others(self.accepted_cars):
                    self.accepted_cars.append(Car(x.item(), y.item()))
                    break

                attempts += 1
                if attempts > self.args.max_attempts:
                    break

        image = torch.FloatTensor(self.to_image()).to(pyprob.util._device)

        scale = torch.tensor(self.args.lik_sigma).to(pyprob.util._device)
        likelihood = Normal(image, scale)
        pyprob.observe(likelihood, name="depth_image_sample",constants={'scale'
                                                                        :
                                                                        scale})
        # TODO: somehow add a "fail" variable for whenever you don't spawn all
        # the cars.
        if generate_samples:
            return number_of_vehicles, image
        else:
            return number_of_vehicles
