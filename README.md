# ProjectTemplate

## Explanations and Instructions

This repository contains the files needed to initialize a project for your [DataScientest](https://datascientest.com/) training.

It contains mainly the present README.md file and an application template [Streamlit](https://streamlit.io/).

**README**

The README.md file is a central element of any git repository. It allows you to present your project, its objectives, and to explain how to install and launch the project, or even how to contribute to it.

You will have to modify different sections of this README.md to include the necessary informations.

- Complete the sections (`## Presentation and Installation` `## Streamlit App`) following the instructions in these sections.
- Delete this section (`## Explanations and Instructions`)

**Streamlit Application**

A [Streamlit] application template (https://streamlit.io/) is available in the [streamlit_app](streamlit_app) folder. You can use this template to start with your project.

## Presentation and Installation

This repository contains the code for our project **Road Accidents in France**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The goal of this project is to model and predict the severity of road accidents in France and identify high-risk factors (areas, human behaviors and circumstances) leading to accidents. This information is invaluable for various stakeholders, including government agencies, law enforcement, and insurance companies. The cleaned dataset is included [here](./data) and the raw data is freely [available](https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/) from the French government.

This project was developed by the following team :

- Pablo Galán de Anta ([GitHub](https://github.com/gdapablo) / [LinkedIn](http://www.linkedin.com/in/pablo-gal%C3%A1n-297075150))
- Jennifer Poehlsen ([GitHub](https://github.com/jpoehlsen) / [LinkedIn](http://linkedin.com/in/jennifer-poehlsen-0aa7a825/))
- Ilinca Suciu ([GitHub](https://github.com/ili-s) / [LinkedIn](http://www.linkedin.com/in/ili-s))

You can browse and run the [notebooks](./notebooks). 

You will need to install the dependencies (in a dedicated environment) :

```
pip install -r requirements.txt
```

## Streamlit App

**Add explanations on how to use the app.**

To run the app (be careful with the paths of the files in the app):

```shell
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).
