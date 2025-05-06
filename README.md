# ProjectTemplate

## Presentation and Installation

This repository contains the code for our project **Road Accidents in France**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The goal of this project is to model and predict the severity of road accidents in France and identify high-risk factors (areas, human behaviors and circumstances) leading to accidents. This information is invaluable for various stakeholders, including government agencies, law enforcement, and insurance companies. The cleaned dataset is included [here](./data) and the raw data is freely [available](https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/) from the French government.

This project was developed by the following team :

- Pablo Galán de Anta ([GitHub](https://github.com/gdapablo) / [LinkedIn](http://www.linkedin.com/in/pablo-gal%C3%A1n-297075150))
- Jennifer Poehlsen ([GitHub](https://github.com/jpoehlsen) / [LinkedIn](http://linkedin.com/in/jennifer-poehlsen-0aa7a825/))
- Ilinca Suciu ([GitHub](https://github.com/ili-s) / [LinkedIn](http://www.linkedin.com/in/ili-s))

You can browse and run the [notebooks](./notebooks). 

You will need to install the dependencies (in a dedicated environment) for running the scripts and the streamlit app. It is highly recommended to use Conda:

For Linux:
```
conda env create -f environment_linux.yml
```

then, you have to activate the environment:
```
conda activate ml_env_linux
```

For Windows:
```
conda env create -f environment_win.yml
```
to activate the environment using conda:
```
ml_env_linux
```
