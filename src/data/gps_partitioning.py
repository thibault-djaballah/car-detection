# -*- coding: utf-8 -*-
import json
import logging
import math
import numpy as np

from pathlib import Path
from dotenv import find_dotenv, load_dotenv

R = 6371*10**3


def get_random_xy(radius, nb_points):

    random_points = np.random.random((nb_points, 2))
    random_points = [(math.sqrt(r), angle * 2 * math.pi) for r, angle in random_points]
    random_points = [(r * math.cos(angle), r * math.sin(angle)) for r, angle in random_points]
    random_points = [(x * radius, y * radius) for x, y in random_points]

    return random_points


def from_xy_to_lat_lng(lat_lng_coord, xy_coord):

    lat, lng = lat_lng_coord
    x, y = xy_coord
    new_lng = x / R * math.cos(lat) * 180 / math.pi + lat
    new_lat = y / R * 180 / math.pi + lng

    return new_lng, new_lat

def get_lat_lng(cities, chosen_country, chosen_city):

    lat, lng = [(float(i['lat']), float(i['lng'])) for i in cities
                if i['country'] == chosen_country and i['name'] == chosen_city][0]
    xy_coords = get_random_xy(10 * 10 ** 3, 100)
    lat_lng_coords = [from_xy_to_lat_lng((lat, lng), xy_coord) for xy_coord in xy_coords]

    return lat_lng_coords


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    cities_filepath = f"{project_dir}/data/external/cities.json"

    with open(cities_filepath) as f:
        cities = json.load(f)

    lat_lng_coords = get_lat_lng(cities, "FR", "Paris")

    print(lat_lng_coords)
