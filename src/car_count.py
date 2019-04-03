import json
import logging
import os

from collections import Counter
from darkflow.net.build import TFNet
from dotenv import find_dotenv, load_dotenv
from multiprocessing import Pool
from pathlib import Path

from src.data.get_image_from_gps import get_lat_lng, GetStreetViewImages
from src.data.get_cars_from_image import get_all_images, get_cars_from_images
from src.models.predict_model import get_car_images_generator, get_maker_detector_model, predict_maker

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[1]

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

    lat_lng_coords = get_lat_lng(cities, country, city, 100, 10 * 10 ** 3)

    pool = Pool(os.cpu_count())  # Create a multiprocessing Pool
    pool.map(GetStreetViewImages(output_dir, GOOGLE_API_KEY), lat_lng_coords)

    input_images = get_all_images(project_dir, country, city)

    output_dir = "{}/data/processed/{}/{}/images".format(project_dir, country, city)

    yolo_v2_dir = "{}/models/yolo_v2".format(project_dir)
    options = {"model": "{}/cfg/yolo.cfg".format(yolo_v2_dir),
               "load": "{}/bin/yolov2.weights".format(yolo_v2_dir),
               "config": "{}/cfg".format(yolo_v2_dir),
               "threshold": 0.1}

    tfnet = TFNet(options)

    for img in input_images:
        get_cars_from_images(tfnet, img, output_dir)

    data_dir_filepath = "{}/data/processed/{}/{}".format(project_dir, country, city)
    model_filepath = "{}/models/mobileNet/mobileNet_mobile_net.35-0.72.hdf5".format(project_dir)

    model = get_maker_detector_model(model_filepath)

    data_generator = get_car_images_generator(data_dir_filepath)

    print(Counter(predict_maker(data_generator, model)))
