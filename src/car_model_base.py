#!/usr/bin/env python3
import numpy as np
import PIL.Image as Image
from pyprob import Model
from util import Car


class BaseCarModel(Model):
    """model."""

    def __init__(self, args):
        """Initialize base model."""
        super().__init__(name="Generative model")

        self.accepted_cars = []

        self.x = np.linspace(-10, 10, 256)
        self.y = np.linspace(-10, 10, 256)

        self.args = args

    def to_image(self, x=None, y=None):
        """Render image."""
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        nx, ny = x.size, y.size
        img = Image.new("RGB", (nx, ny))

        for car in self.accepted_cars:
            car.draw_img(x, y, img)

        return np.array(img).astype('float')[:, :, 0]

    def attempt_to_place_car(self, x, y, rot=0):
        """Attempt to place car, return True if succesful, False otherwise."""
        newcar = Car(x.item(), y.item(), rot=rot)
        if not newcar.overlaps_with_others(self.accepted_cars):
            self.accepted_cars.append(newcar)
            return True
        else:
            return False

    def reset_model(self):
        """Reset."""
        self.accepted_cars = []
