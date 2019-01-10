import keras
import logging
import numpy as np

from collections import Counter
from dotenv import find_dotenv, load_dotenv
from pathlib import Path


def get_maker_detector_model(model_directory):
    return keras.models.load_model(model_directory)


def get_car_images_generator(car_images_directory):

    data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    predictions = data_generator.flow_from_directory(car_images_directory, target_size=(224, 224), batch_size=32)

    return [list(data_generator.class_indices.keys())[i] for i in predictions]


def predict_maker(data_generator, model):

    predictions = model.predict_generator(data_generator, steps=data_generator.samples / 128)

    predictions = [np.argmax(pred) for pred in predictions]

    return predictions


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

    data_dir_filepath = "{}/data/processed/{}/{}".format(project_dir, country, city)
    model_filepath = "{}/model/resnext50/lol.h5".format(project_dir)

    model = get_maker_detector_model(model_filepath)

    data_generator = get_car_images_generator(data_dir_filepath)

    print(Counter(predict_maker(data_generator, model)))




