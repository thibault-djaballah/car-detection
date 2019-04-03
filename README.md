### car-detection project


![Screenshot](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/illustration.png)
<br/> The car-detection project tries to solve a useless but essential
mission:

What are the market shares of the automotive makers in a city ?

Based on Google Street View pictures and neural networks, the output is a count of each brand detected on the street.

### Requirements

Requirements are in the requirements.txt file.
 
### Methodology

**Step 1:**<br/> From a city lat, long we scrape randomly N images
from Google Street View.

**Step 2:**<br/>
From these N randomly chosen images we detect cars on it thanks to pre-trained YOLOV2 and then crop it.

**Step 3:**<br/>
We put those cropped car images into a hand-made resneXt to predict the car maker.

**Step 4:**<br/> We finally obtain a car count grouped by maker.

**NB :** <br/> This car-maker-classifier (resneXt architecture) was
trained on
[CompCars Dataset](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)
</a>. <br/> For now car makers are just id integers. The corresponding
mapping with the name of the maker is in CompCar Dataset. <br/> You will
need a GOOGLE API KEY in the dotenv file to query the google streeet
view api.

### Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │               
    │      
    │   
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
