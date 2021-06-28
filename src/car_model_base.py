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
from util import Car


class BaseCarModel(Model):
    """model"""
    def __init__(self, args):
        super().__init__(name="Generative model")

        self.accepted_cars = []

        self.x = np.linspace(-15, 15, 1000)
        self.y = np.linspace(-15, 15, 1000)

        self.args = args

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
