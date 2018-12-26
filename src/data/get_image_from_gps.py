# -*- coding: utf-8 -*-
import json
import logging
import numpy as np
import os
import google_streetview.api
import uuid

from dotenv import find_dotenv, load_dotenv
from multiprocessing import Pool
from pathlib import Path

from src.data.gps_partitioning import get_lat_lng


def get_image_from_lat_lng(output_dir, google_api_key, lat_lng_coord):
    lat, lng = lat_lng_coord
    # Define parameters for street view api
    params = [{
        'size': '640x640',  # max 640x640 pixels
        'location': '{},{}'.format(lat, lng),
        'heading': '{}'.format(np.random.randint(0, 360)),
        'pitch': '{}'.format(np.random.randint(-20, 0)),
        'key': '{}'.format(google_api_key)
    }]

    # Create a results object
    results = google_streetview.api.results(params)
    output_filepath = '{}/{}'.format(output_dir, uuid.uuid4())
    # Download images to directory 'downloads'
    print('Downloading {})'.format(output_filepath))
    results.download_links(output_filepath)


class GetStreetViewImages(object):

    def __init__(self, output_dir, google_api_key):
        self.output_dir = output_dir
        self.google_api_key = google_api_key

    def __call__(self, lat_lng_coord):
        get_image_from_lat_lng(self.output_dir, self.google_api_key, lat_lng_coord)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

    cities_filepath = "{}/data/external/cities.json".format(project_dir)

    with open(cities_filepath) as f:
        cities = json.load(f)

    country = "FR"
    city = "Bordeaux"
    output_dir = '{}/data/raw/{}/{}'.format(project_dir, country, city)

    lat_lng_coords = get_lat_lng(cities, country, city, 50, 10 * 10 ** 3)

    pool = Pool(os.cpu_count())  # Create a multiprocessing Pool
    pool.map(GetStreetViewImages(output_dir, GOOGLE_API_KEY), lat_lng_coords)

