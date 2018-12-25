# -*- coding: utf-8 -*-
import logging

from darkflow.net.build import TFNet
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def get_image_from_lat_lng(output_file_path, lat, lng, google_api_key):

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    country = "FR"
    city = "Bordeaux"

    # Get our yolo network
    yolo_v2_dir = "{}/models/yolo_v2".format(project_dir)
    options = {"model": "{}/cfg/yolo.cfg".format(yolo_v2_dir),
               "load": "{}/bin/yolov2.weights".format(yolo_v2_dir),
               "threshold": 0.1}

    tfnet = TFNet(options)
