#!/usr/bin/env python3
import numpy as np
from shapely.geometry import Polygon, mapping
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import pyprob
from pyprob import Model
from pyprob.distributions import Normal, Uniform, Poisson
from types import SimpleNamespace
import matplotlib.pyplot as plt
import torch


class Car():
    def __init__(self, x, y, rot=0):
        self.x = x
        self.y = y

        self.rot = rot

        self.length = 1.
        self.width = 1.

        self.xy = np.array([[- self.length/2, -self.width/2],
                            [self.length/2, -self.width/2],
                            [self.length/2, self.width/2],
                            [- self.length/2, self.width/2]])

        self.xy = np.matmul(rot_z(self.rot).T, self.xy.T).T

        self.xy[:, 0] += self.x
        self.xy[:, 1] += self.y

        self.poly = Polygon(self.xy)

    def overlaps_with_others(self, other_cars):
        for other_car in other_cars:
            if self.overlaps_with_other(other_car):
                return True
        return False

    def overlaps_with_other(self, other_car):
        if self.intersect(other_car) > 0.:
            return True

    def intersect(self, other_car):
        """Calc intersection with other car"""

        area_overlap = self.poly.intersection(other_car.poly).area
        return area_overlap

    def draw_img(self, x,y, img):
        x_pixels = np.digitize(self.xy[:,0],x)
        y_pixels = np.digitize(self.xy[:,1],y)
        draw = ImageDraw.Draw(img)

        points = np.array([x_pixels, y_pixels]).T

        points = tuple([tuple(pt) for pt in points])

        draw.polygon((points), fill=1)


def rot_z(rot):
    """Rotate in degrees"""
    return np.array([[np.cos(rot*np.pi/180), -np.sin(rot*np.pi/180)],
                     [np.sin(rot*np.pi/180), np.cos(rot*np.pi/180)]])

def map_accept(x, y):
    """This function takes the role of a "map", here it is just limited to a
    radius of 50. """
    limit = 50.
    if torch.sqrt(x**2 + y**2) > limit:
        return False
    else:
        return True


class CarModel(Model):
    """model"""
    def __init__(self, args):
        super().__init__(name="Generative model")

        self.accepted_cars = []

        self.x = np.linspace(-15, 15, 1000)
        self.y = np.linspace(-15, 15, 1000)

        self.args = args

        self.args.spread = 5.
        self.args.max_attempts = 100
        self.args.lik_sigma = 0.05



    def to_image(self, x=None, y=None):

        if x is None:
            x = self.x
        if y is None:
            y = self.y

        nx,ny = x.size,y.size
        img = Image.new("RGB", (nx, nx))

        for car in self.accepted_cars:
            car.draw_img(x, y, img)

        return np.array(img).astype('float')[:,:,0]


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
