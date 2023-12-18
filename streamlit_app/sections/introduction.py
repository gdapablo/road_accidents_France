import streamlit as st, pandas as pd
from matplotlib.pyplot import imread

def section1():
    st.image("assets/traffic.png")
    image_link = f'<a href=https://auto.hindustantimes.com/auto/news/traffic-jam-as-long-as-700-kms-paris-chokes-on-day-one-of-fresh-lockdown-41604116978417.html> Image credit: Bloomberg'
    st.caption(image_link, unsafe_allow_html=True)

    st.write("## Introduction & pre-processing")

    st.write("### Authors: Ilinca Suciu, Jennifer Poehlsen and Pablo M. Galán de Anta")
    st.write('The goal of the present project is to predict the severity of road accidents \
    in France from official data provided by the French government. One novelty aspect of our \
    analysis consists in the integration of the newly released datasets for 2022 (as of Oct. 5, 2023).')

    st.write('### Data Presentation')
    #<<<<<<< HEAD
    df = pd.read_csv('./tables/accident_data_head_10.csv', index_col = 0)
    df['severity'].replace({1:0,2:1,3:2,4:3}, inplace=True)
    st.dataframe(df)
    #=======
    #    st.dataframe(merge.head(10))
    #>>>>>>> d37ccaa01f7018d6981e0208168fe54af8c8869e
    st.write('The target variable is severity, which shows information about the \
    injury severity of the accident for each person involved, called users. Users can be a driver, passenger or pedestrian.')
    st.write('The vast majority of categories are categorical, with multiple categories. The exceptions are \
    the variables year and age which are quantitative.')

    st.write('#### Data shape')
    st.write('(2665528, 57)')
    st.write('The data was collected from 2005 till 2022.')

    # Define the schema
    schema_data = {
        'Severity Code': [0, 3, 2, 1],
        'Injury Severity': ['Unharmed', 'Minor injury', 'Hospitalized injury', 'Killed']
    }

    # Create a DataFrame from the schema data
    schema_df = pd.DataFrame(schema_data)

    # Display the schema in a Streamlit app
    st.write("### Injury Severity Schema")
    st.write('The initial schema was:')
    st.table(schema_df)

    st.write('The order was odd as it was not showing a logical order so we re-arranged it')
    # Define the schema
    schema_data = {
        'Severity Code': [0, 1, 2, 3],
        'Injury Severity': ['Unharmed', 'Minor injury', 'Hospitalized injury', 'Killed']
    }

    # Create a DataFrame from the schema data
    schema_df = pd.DataFrame(schema_data)
    st.table(schema_df)

    #length = len(merge)

    st.write('#### Data statistics and nulls')
    df = pd.read_csv('./tables/accident_data_stats.csv', index_col = 0)
    st.dataframe(df)

    if st.checkbox("Show NA in percentages"):
        df = pd.read_csv('./tables/accidents_data_percent_NA.csv', index_col = 0)
        st.dataframe(df)
    st.write('There are multiple columns with NAs. The variables secu1, secu2, and secu3 were collected \
    since 2019, so for the remaining years we do not have data.')

    st.write("### Pre-processing")
    st.write('We start by renaming columns \'an\', \'mois\', and \'jour\' along with the categories in injury severity which are not sorted.')
    st.write('We also put the column \'an\' (year) in a logical format 20XX and create a new column \'age\' which includes the age of the \
    user involved in the accident (driver, passenger or pedestrian).')

    # Also substitute the Severity to be sortened
    severity_mapping = {
        1: 1,
        2: 4,
        3: 3,
        4: 2
    }

    st.write('Due to the high percentage of NAs, we can delete columns with more than 90% NAs.')

    st.write('As most columns are categorical, we can convert NAs into an \'Unknown\' category such as -1. \
    In this way, we can convert categorical variables into integers.')
    st.write('Some variables like \'actp\' contains 0.0 and 0 values that require to be merged into a single one.')
    df = pd.read_csv('./tables/accident_data_actp.csv', index_col = 0)
    st.dataframe(df)
    st.write('Some entries like \'A\' or \'B\' can be converted into new numerical variables like -2 and -3.')

    st.write('#### Outliers in \'age\'')

    st.write('If we take a look at the distribution of ages by injury severity and age of the user, we find aberrant \
    values in the \'age\' column:')

    # Load the image
    im = imread('../plots/boxplot_age.png')

    # Display the image without labels using st.image
    st.image(im, use_column_width=True, output_format='PNG')

    st.write('So if we delete aberrant values and we take a reasonable minimum age for drivers (18 years old in France) \
    we get:')

    # Load the image
    im = imread('../plots/boxplot_age_cleaned.png')

    # Display the image without labels using st.image
    st.image(im, use_column_width=True, output_format='PNG')

    st.write('#### Aberrant values in \'long\' and \'lat\' columns')

    st.write('In the variables \'long\' and \'lat\' there are aberrant values in invalid formats. There are also longitudes and latitudes \
    not corresponding to mainland France. These could be overseas French territories or wrong values. We exclude these values as we are interested in \
    analysing the accidents in mainland France.')

    # Load the image
    im = imread('../plots/long_lat.png')

    # Display the image without labels using st.image
    st.image(im, use_column_width=True, output_format='PNG')
