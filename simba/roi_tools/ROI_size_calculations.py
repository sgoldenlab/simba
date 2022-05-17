import numpy as np
import math

def rectangle_size_calc(rectangle_dict, px_mm):
    rectangle_dict['height_cm'] = round((rectangle_dict['height'] / px_mm) / 10, 2)
    rectangle_dict['width_cm'] = round((rectangle_dict['width'] / px_mm) / 10, 2)
    rectangle_dict['area_cm'] = round(rectangle_dict['width_cm'] * rectangle_dict['height_cm'], 2)

    return rectangle_dict

def circle_size_calc(circle_dict, px_mm):
    radius_cm = round((circle_dict['radius'] / px_mm) / 10, 2)
    circle_dict['radius_cm'] = radius_cm
    circle_dict['area_cm'] = round(math.pi * (radius_cm ** 2), 2)

    return circle_dict

def polygon_size_calc(polygon_dict, px_mm):
    y_vals = polygon_dict['vertices'][:,0]
    x_vals = polygon_dict['vertices'][:,1]
    poly_area_px = 0.5 * np.abs(np.dot(x_vals, np.roll(y_vals, 1)) - np.dot(y_vals, np.roll(x_vals, 1)))
    polygon_dict['area_cm'] = round((poly_area_px / px_mm) / 500, 2)

    return polygon_dict