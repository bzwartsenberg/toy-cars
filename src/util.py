#!/usr/bin/env python3
import numpy as np
from shapely.geometry import Polygon, mapping
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image


class Car():
    """Car class."""
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

