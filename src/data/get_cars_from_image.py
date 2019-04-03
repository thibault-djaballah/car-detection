# -*- coding: utf-8 -*-
import cv2
import glob
import logging
import uuid

from darkflow.net.build import TFNet
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def get_cars_from_images(tfnet, image_file_path, output_dir):

    imgcv = cv2.imread(image_file_path)
    result = tfnet.return_predict(imgcv)
    result = [d for d in result if d['confidence'] > 0.5 and d['label'] == 'car']

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for d in result:
        topleft = d['topleft']
        bottomright = d['bottomright']

        xDiff = abs(topleft['x'] - bottomright['x'])
        yDiff = abs(topleft['y'] - bottomright['y'])

        topleft['y'] = max(int(topleft['y'] - 0.1 * yDiff), 0)
        topleft['x'] = max(int(topleft['x'] - 0.1 * xDiff), 0)

        bottomright['y'] = max(int(bottomright['y'] + 0.1 * yDiff), 0)
        bottomright['x'] = max(int(bottomright['x'] + 0.1 * xDiff), 0)

        output_filepath = '{}/{}.jpg'.format(output_dir, uuid.uuid4())

        cv2.imwrite(filename=output_filepath,
                    img=imgcv[topleft['y']:bottomright['y'], topleft['x']:bottomright['x']])

    return result


def get_all_images(project_dir, country, city):
    return glob.glob('{}/data/raw/{}/{}/*/*.jpg'.format(project_dir, country, city))


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
