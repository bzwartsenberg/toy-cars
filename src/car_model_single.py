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

        number_of_vehicles = 1

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
        if generate_samples:
            return number_of_vehicles, image
        else:
            return number_of_vehicles


if __name__ == '__main__':

    pyprob.set_random_seed(42424242)

    args = SimpleNamespace()


    model = CarModel(args)

    number_of_vehicles, image = model.get_trace(generate_samples=True, verbose=True).result

    fig,ax = plt.subplots(figsize=(10,10))

    print('Spawned number of vehicles: ', number_of_vehicles)

    ax.pcolormesh(model.x, model.y, image.detach().numpy(), shading='auto')
    plt.show()

    plt.savefig('img/test.png')
