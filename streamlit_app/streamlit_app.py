#!/usr/bin/env python3
import glob
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import seaborn as sns
import joblib, glob, tensorflow as tf
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, precision_recall_curve, classification_report, recall_score
from sklearn.feature_selection import VarianceThreshold
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
import datetime
import statsmodels.api as sm
from itertools import product

import config
from member import Member

@st.cache_data
def load_data(file_pattern):
    files = glob.glob(file_pattern)
    files = sorted(files)
    data_parts = [pd.read_csv(file_path) for file_path in files]
    return pd.concat(data_parts, ignore_index=True)

# Openning the merged DataFrame
# merge = load_data('../data/merge_data/merge_parts_*')

# We can delete columns id_vehicule, id_usager and num_veh as they are pointless for the analysis and
# the identificator of each accident can be extracted from Num_Acc
# cols_to_delete = ['id_usager', 'id_vehicule_x', 'id_vehicule_y']
# merge.drop(columns=cols_to_delete, axis=1, inplace=True)

st.title("Road Accidents in France: SEP23 Bootcamp project")
st.sidebar.title("Table of contents")
pages=["Introduction & pre-processing",
"Data Visualization", "Modelling", 'Time Series', 'Conclusions']
page=st.sidebar.radio("Go to", pages)

st.sidebar.markdown("---")
st.sidebar.markdown(f"## {config.PROMOTION}")
st.sidebar.markdown("### Team members:")
for member in config.TEAM_MEMBERS:
    st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)


# ==================================
# First page: Data Exploration
if page == pages[0]:

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

# ==================================

# ==================================
# Second page: Pre-processing and Visualization
if page == pages[1]:
    st.write("## Data Visualization")

    st.write('Making use of the pre-processed data, we take a look at different variables to look for correlations or some \
    valuable information we can extract by relating the different variables in the dataset.')

    st.write('### Seat occupied in crashed cars - by gender')
    # Load the image
    im = imread('../plots/position_in_car_crash_by_gender_streamlit.png')

    # Display the image without labels using st.image
    st.image(im, use_column_width=True, output_format='PNG')
    st.write('In France, the front left place corresponds to the driver seat, and we observe a disparity in the 2 distributions. \
             Women, compared to men, were less frequently behind the wheel of cars involved in accidents. One big factor contributing \
             to this effect may be the gender gap in the possession of driver license (although in some countries - like the U.S. - \
             that gap was closed and even reversed). Interestingly, the place next to the driver (front right) is more frequently \
             occupied by women, whereas the 2 back seats have a slight tendency to be occupied more often by men.')

    st.write('### Number of accidents per day')
    # Load the image
    im = imread('../plots/num_acc_per_day_streamlit.png')

    # Display the image without labels using st.image
    st.image(im, use_column_width=True, output_format='PNG')
    st.write('There is a clear increase in the number of accidents that occur on Fridays compared to the rest of the week. \
              Sundays, on the other hand, have visibly fewer accidents than the other days of the week.')

    st.write('### Number of accidents per hour')
    # Load the image
    im = imread('../plots/num_acc_per_hour_streamlit.png')

    # Display the image without labels using st.image
    st.image(im, use_column_width=True, output_format='PNG')
    st.write('There are fewer accidents during the day than at night. The number of accidents spikes during rush hour, \
              with there being significantly more accidents in the evening rush hour than in the morning.')

    st.write('### Accidents by age group')
    # Load the image
    im = imread('../plots/severity_by_age-groups_streamlit.png')

    # Display the image without labels using st.image
    st.image(im, use_column_width=True, output_format='PNG')
    st.write('Although the number of fatalities remains relatively low compared to other injury categories, it peaks among \
             pedestrians aged 66-80 years, indicating their heightened vulnerability in accidents. This contrasts sharply with \
             the trend among drivers, where the proportion of unharmed individuals increases with age. Among drivers, the oldest \
             individuals are the least likely to sustain minor injuries, with a significant majority emerging unharmed from accidents. \
             In contrast, the youngest drivers are more prone to minor injuries and hospitalization.')

    st.write('### Casualties over years relative to 2005')
    # Load the image
    im = imread('../plots/casualties_over_time_2005_streamlit.png')

    # Display the image without labels using st.image
    st.image(im, use_column_width=True, output_format='PNG')
    st.write('All 4 injury categories follow the same trend between 2010 and 2017: the relative injury numbers decrease at a \
             constant slope until 2013, when they reach a plateau relatively stable until 2017. After 2017, the trendlines \
             diverge: there is a steep decrease in the % of hospitalized patients, whereas the % death toll and % unharmed don\’t \
             change much until 2019. Over these 2 years, the drop in hospitalizations is mirrored by an increase in % minor injuries. \
             In 2020, the decline is visible for all categories. In the following year, the numbers increased again. They reach values \
             close to those from the year preceding the SARS-Cov2 pandemic, except for the number of hospitalizations, which in 2022 \
             are still below those from 2019. The % unharmed and % number of accidents curves show an almost perfect overlap.')

    st.write('### Heatmap of accident fatalities for year 2022')
    # Load the image
    p = open("../plots/2022_map_with_colored_markers_final.html")
    description = (
        "Explore a heatmap of all road accident fatalities in mainland France for the year 2022. The map is visualized using geographical coordinates (latitude and longitude). "
        "When zoomed out, fatalities are aggregated per view area. "
        "<span style='color: orange;'>Zoom in</span> to view individual fatalities. "
        "<span style='color: orange;'>Click on the markers</span> to reveal additional information about the victim and accident circumstances, including the date of the accident, the victim's gender, age, and role (driver, passenger, pedestrian)."
        )
    st.markdown(description, unsafe_allow_html=True)
    components.html(p.read(), height=450)
# ==================================

# ==================================
# Third page: Modelling
if page == pages[2]:
    st.write("## Modelling")

    st.write('When conducting machine learning modeling, our emphasis is on the years 2019-2022, as they offer high-quality \
    data with fewer instances of missing values (NaNs). This time frame provides a substantial number of entries, ensuring the \
    potential for achieving robust results while remaining computationally feasible for performance on a standard personal computer (PC).')

    st.write('In the preprocessing stage, we simplify the classification of injury severity by condensing the four original categories into two: \
    categorizing \'0\' as \'unharmed\' and \'1\' as the grouping for every user involved in the accident who suffered injury or fatality.')

    # Load the image
    im = imread('../plots/injury_sev_2cats.png')

    # Display the image without labels using st.image
    st.image(im, use_column_width=True, output_format='PNG')

    st.write('__Figure 1.__ _Distribution of the data with a binary severity. A severity of 0 means unharmed, a severity of 1 means injured \
    (minor or hospitalization) or killed._')

    st.write('### Metric selection: Recall')

    st.write('We are interested in predicting injury severity with two categories: 0 for unharmed and 1 for killed/injured. \
    Given this scenario, focusing on the recall metric is justified because of two main reasons:')

    text = """

    - __Importance of Identifying Positive Cases:__ In the context of road accidents, identifying cases where individuals are \
    killed or injured is particularly important. The consequences of missing these cases (false negatives) can be severe, as it \
    may lead to insufficient safety measures or interventions. Maximizing recall helps ensure that the model is effective in identifying \
    instances of the positive class.

    - __Public Health Impact:__ Minimizing the number of severe injuries or fatalities in road accidents is a critical goal. \
    Maximizing recall helps to identify and address the instances that lead to serious consequences, contributing to overall safety improvements.

    """

    st.markdown(text)

    st.write('### Features importance')

    st.write('We reduce the dimensionality of the problem by performing a features importance analysis using \
    a logistic regression. First, we employ a random forest to get an array of the importance per feature, and \
    then, by taking different importance thresholds, we run logistic regressions on every importance threshold to \
    capture the evolution of precision, recall, and F1-score and decide  on the number of features that preserves the \
    scores while reducing the dimension for better computational times.')

    # Load the image
    im = imread('../plots/dimension_reduc.png')

    # Display the image without labels using st.image
    st.image(im, use_column_width=True, output_format='PNG')

    st.write('__Figure 2.__ _Score of precision, recall and F1-score for logistic regression using different numbers \
    of features accounting for their importance._')

    st.write('From the plot, we can deduce there is a plateau down to ~100 features and then the scores smoothly decrease.')

    # Combine the logic for loading training and testing files
    @st.cache_data
    def load_data(file_pattern):
        files = glob.glob(file_pattern)
        files = sorted(files)
        data_parts = [pd.read_csv(file_path) for file_path in files]
        return pd.concat(data_parts, ignore_index=True)

    @st.cache_data
    def load_model():
        model_path = '../models/DNN_model.keras'
        if isinstance(model_path, dict):
            model_path = model_path[ini]

        import io, sys
        model = tf.keras.models.load_model(model_path)

        # Redirect sys.stdout to capture the summary
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        # Print the model summary
        model.summary()

        # Reset sys.stdout
        sys.stdout = old_stdout

        # Get the captured summary as a string
        model_summary = new_stdout.getvalue()

        # Display the captured summary
        st.write("#### Model Summary")
        st.text(model_summary)

        return model

    models = ['LR', 'RFC', 'XGboost', 'DNN(>20%)', 'DNN(>30%)']
    recall = [0.7848, 0.82, 0.8337, 0.95, 0.9014]
    prec   = [0.8083, 0.8105, 0.7427, 0.703, 0.7551]

    st.write('### Models performance')

    st.write('We conducted an extensive analysis on the 2019-2022 dataset, employing four distinct \
    machine learning models to predict instances of users being injured or killed in accidents. Through \
    meticulous grid search cross-validation, we identified optimal hyper-parameter configurations for each \
    model, emphasizing the maximization of recall. The models employed include Logistic Regression (LR), \
    Random Forest Classifier (RFC), XGBoost, and Dense Neural Network (DNN). The comprehensive evaluation \
    of their performance, specifically in terms of recall and precision, is visually presented in Figure 3.')

    # Plotting the bar plot
    fig, ax = plt.subplots()

    # Bar plot for recall values
    bars = ax.bar(models, recall, color='blue')

    # Adding precision values as color to the bars
    for bar, precision_value in zip(bars, prec):
        bar.set_color(plt.cm.RdYlBu(precision_value))

    # Setting labels and title
    ax.set_ylabel('Recall', color='white')
    ax.set_title('Recall and Precision for Injured/Killed Category', color='white')
    ax.set_ylim(0,1)

    # Adding color bar
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu)
    mappable.set_array(prec)
    mappable.set_clim(0.75, 0.85)  # Set the color bar range between 0.7 and 0.8
    cbar = plt.colorbar(mappable, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label('Precision', color='white')

    fig.patch.set_facecolor('#2c3e50')

    # Rotate x-axis ticks
    ax.set_xticklabels(models, rotation=45)  # Corrected line

    # Set the color of tick labels on the color bar to white
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.yaxis, 'ticklabels'), color='white')

    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')

    st.pyplot(fig)

    st.write('__Figure 3.__ _Recall of injured/killed category for four machine learning models, from left \
    to right: Linear Regression, Random Forest Classifier, XGboost, DNN considering every probability \
    above 20% to be injured/killed, and DNN with probability above 30%. The colorbar shows the \
    precision score of each injured/killed model category._')

    st.write('### A particular case: Dense Neural Network')

    st.write('In optimizing the Dense Neural Network (DNN), we experimented with varying configurations of \
    hidden layers and neurons per layer to enhance the recall score. Given the binary nature of the \
    classification task (unharmed or injured/killed), the output is structured as a list of probabilities \
    ranging from 0 to 1. This is achieved through a singular neuron in the output layer, utilizing a \
    sigmoid activation function. The refined architecture of the DNN, representing the optimal configuration, \
    is outlined in the subsequent table.')

    # Load training and testing data
    training_set = load_data('../data/train_test_sets/training_set_parts_*')
    testing_set  = load_data('../data/train_test_sets/testing_set_parts_*')

    # Splitting in Features and Target
    X_train, y_train = training_set.drop('severity', axis=1), training_set['severity']
    X_test, y_test = testing_set.drop('severity', axis=1), testing_set['severity']

    # Open the chosen model
    clf = load_model()

    st.write('In this model, we leverage the flexibility to fine-tune the classification threshold based on \
    output probabilities. Specifically, our emphasis is on predicting category 1 (injured/killed), prompting \
    us to adopt a conservative approach with the probabilities. This adjustment allows us to be more stringent \
    in predicting category 1, thereby achieving a high recall for instances in this category while maintaining \
    a lower recall for category 0. This strategic threshold manipulation aligns with our objective of \
    accurately identifying cases with more severe outcomes.')

    # Load the image
    im = imread('../plots/DNN_ROC_curve.png')

        # Display the image without labels using st.image
    st.image(im, use_column_width=True, output_format='PNG')

    st.write('__Figure 4.__ _ROC curve (left panel) and precision vs recall curve (right panel) for the DNN. \
    The performance of the DNN is given by the area under the ROC curve, in this case 0.88._')

    st.write('The output of the DNN is a list of probabilities that we might compare with \
    the testing set. We can select the threshold of the probability for a user to be considered as injured/killed.')
    prob = st.number_input('Select the threshold (between 0 and 1):',
    min_value=0.1, max_value=0.9, step=0.05)

    @st.cache_data
    def prediction_dnn(_clf, X_test):
        return clf.predict(X_test)

    y_pred = prediction_dnn(clf, X_test)
    y_pred_new = (y_pred>=prob)
    y_pred_new = y_pred_new[:,0]

    target_names = ["Unharmed", "Injured/killed"]
    st.write('Classification report:')
    st.dataframe(classification_report(y_test, y_pred_new,
    target_names=target_names, output_dict=True))

    st.markdown(
        """
        ### Modelling Conclusions

        - The Deep Neural Network (DNN) stands out as the model with the highest recall, maintaining a \
        commendable level of accuracy in predicting the injured/killed category.

        - Additionally, XGBoost demonstrates impressive performance, surpassing the DNN in computational \
        speed, making it an efficient alternative.

        - Analysis of the 2019-2022 dataset reveals its adequacy for accurately predicting cases of injury \
        or fatality while mitigating the high computational costs associated with utilizing the entire dataset \
        spanning 2005-2022.
        """
    )
# ==================================

# ==================================
# Fourth page: Time Series
if page == pages[3]:
    # Year selector
    @st.cache_data
    def year_selector(year):
        import glob
        list = glob.glob('../data/accidents_*.csv')
        list = sorted(list)
        list_of_years = []
        for jj in list:
            list_of_years.append(int(jj[18:22]))
        list_of_years = np.array(list_of_years)
        file_list = np.array(list)[list_of_years >= year]
        concatenated_df = pd.concat([pd.read_csv(file_path) for file_path in file_list], ignore_index=True)

        return concatenated_df

    st.write("## Time series")

    # Year selector
    choice = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
    2018, 2019]
    option = st.selectbox('Select the year from which you want to start computing the time series:', choice)
    st.write('The year selected is:', option)
    df = year_selector(option)

    df['year'] = df['year'].astype('int64').astype('str')
    df['month'] = df['month'].astype('int64').astype('str')
    df['day'] = df['day'].astype('int64').astype('str')

    df['yymm'] = df['year'] + '-' + df['month'].str.zfill(2)
    df['date'] = df['yymm'] + '-' + df['day'].str.zfill(2)

    df.drop('yymm', axis=1, inplace=True)

    df_to_pred = df[df['date'] >= '2022-12-01'] # Storing last period of 2022 in a new variable
    df = df[df['date'] < '2022-12-01'] # Quiting the last period of 2022 for prediction

    accidents = df.groupby('date')['Num_Acc'].nunique()
    del df

    # Convert the index to a DateTimeIndex
    accidents.index = pd.to_datetime(accidents.index)

    # Same for the accidents to predict
    acc_to_pred = df_to_pred.groupby('date')['Num_Acc'].nunique()
    del df_to_pred

    acc_to_pred.index = pd.to_datetime(acc_to_pred.index)

    st.write("### Data presentation")
    st.dataframe(accidents.head(5))

    # Plotting the number of accidents
    st.write('Number of accidents per date')
    fig, ax = plt.subplots(1,1,figsize=(10,4))
    ax.plot(accidents)
    ax.set_xlabel('Date', color='white')
    ax.set_ylabel('# Accidents', color='white')

    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')

    fig.patch.set_facecolor('#2c3e50')

    st.pyplot(fig)

    st.write("The data appears to be stationary. The Augmented Dickey-Fuller Test (ADF) gives a p-value of 4e-9, which is consistent with stationary.")

    fig = plt.figure()
    pd.plotting.autocorrelation_plot(accidents)
    st.pyplot(fig)
    st.write("The autocorrelation plot does tend to zero, but has significant seasonal peaks.")

    # Seasonality selector
    st.write('As we saw a significant dependence of the number of accidents on the day of the week, we expect a seasonality of S=7 (7 days).')
    season = 7

    # We apply the seasonal_decompose function to accidents
    accidentslog = np.log(accidents)

    #We have a seasonality of period 7 months
    accidents_ma = accidentslog.rolling(window = season, center = True).mean()

    # Here we use the transform in log we are therefore in an additive model

    mult = seasonal_decompose(accidentslog)

    # Seasonal coefficients are subtracted from the accidentslog series
    cvs=accidentslog - mult.seasonal

    # We go to the exponential to find the original series
    x_cvs=np.exp(cvs)

    # We display the series
    #st.write('### Corrected time series and moving average')
    fig, ax = plt.subplots(1,2,figsize=(10,4),sharey=True)
    ax = ax.ravel()

    ax[0].plot(accidents, label='Original series')
    ax[0].plot(x_cvs, label='Corrected series')

    ax[0].set_title('Graph of the original series and the corrected series', color='white')
    ax[0].set_xlabel('Date', color='white'); ax[1].set_xlabel('Date', color='white')
    ax[0].set_ylabel('Number of passengers', color='white')
    ax[0].legend()

    ax[1].plot(np.exp(accidentslog), color = 'blue', label = 'Origin')
    ax[1].plot(np.exp(accidents_ma), color = 'red', label = 'Moving average')

    ax[1].legend()
    ax[1].set_title('Moving average', color='white')

    ax[0].tick_params(axis='y', colors='white')
    ax[0].tick_params(axis='x', colors='white'); ax[1].tick_params(axis='x', colors='white')

    fig.patch.set_facecolor('#2c3e50')

    plt.tight_layout()
    #st.pyplot(fig)

    # Double differencing process
    #accidentslog_1 = accidentslog.diff().dropna() # 1st order differencing
    #accidentslog_2 = accidentslog_1.diff(periods = season).dropna() # 7 order difference
    accidentslog_2 = accidentslog.diff(periods = season).dropna() # 7 order difference

    # SARIMA model
    st.write('### SARIMA model')
    st.write("We will train a SARIMA model, which will have parameters (p, d, q)(P, D, Q, k), where the seasonality k = 7 .")

    # Plot on 36 lags the simple and partial autocorrelograms of the doubly differenced time series
    st.write('### Simple and partial autocorrelograms')
    st.write("We can get an estimate of the parameters by looking at the simple and partial autocorrelograms.")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), sharey=True)

    plot_acf(accidentslog_2, lags = 36, ax=ax1)
    plot_pacf(accidentslog_2, lags = 36, ax=ax2)

    ax1.tick_params(axis='y', colors='white'); ax1.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white'); ax2.tick_params(axis='x', colors='white')

    ax1.set_title('Autocorrelation', color='white'); ax2.set_title('Partial autocorrelation', color='white')

    fig.patch.set_facecolor('#2c3e50')

    plt.tight_layout()
    st.pyplot(fig)

    st.write("A good first guess for the parameters would be (1, 0, 1), (1, 1, 1, 7).")

    st.write('#### Finding the best parameters')
    st.write('The SARIMA model is fit over a range of (p, d, q)(P, D, Q, S) parameters \
    to get the best fit of all of them, defined by the lowest Akaike\'s Information Criterion (AIC). For this particular time series, the optimal parameters \
    to get the best fit with all order and seasonal order parameters with high significance is shown below.')

    @st.cache_data
    def sarimax_calculation():
        model=sm.tsa.SARIMAX(accidentslog,order=(1,0,2),
        seasonal_order=(1,1,1,season))
        results=model.fit()
        return results

    st.write('Best order: ',(1,0,2))
    st.write('Best seasonal order: ',(1,1,1,7))
    # Display the summary in Streamlit
    results = sarimax_calculation()
    st.text(results.summary())

    st.write('As we can see, all p-values are far below 0.05, exhibiting high relevance for all the parameters. \
    The Ljung-Box test gives a p-value of 0.97, so we can not reject the hypothesis that the residuals are white noise.\
    The JB test shows a probability of 0, then concluding that the residue does not follow a normal distribution.')
    st.write('With the SARIMA model calculated, we can make predictions about the number of accidents.')

    prediction = results.get_forecast(steps=31).summary_frame()  # Forecasting with a confidence interval

    st.write('#### Prediction of number of accidents per date')
    fig, ax = plt.subplots(figsize = (15,5))

    plt.plot(accidents)
    plt.plot(acc_to_pred, 'r--', alpha=0.5, label='Last period data')
    prediction = np.exp(prediction) # Exponential Transform

    prediction['mean'].plot(ax = ax, style = 'k--', label='Prediction') # Plotting the mean

    ax.fill_between(prediction.index, prediction['mean_ci_lower'], prediction['mean_ci_upper'], color='k', alpha=0.1); #Plotting the confidence interval
    ax.set_xlim('2022-6-01','2022-12-31')
    ax.set_title('Accidents prediction', color='white')
    ax.set_xlabel('Date', color='white'); ax.set_ylabel('# Accidents', color='white')
    ax.legend(loc=2)

    ax.tick_params(axis='y', colors='white'); ax.tick_params(axis='x', colors='white')

    fig.patch.set_facecolor('#2c3e50')

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(
        """
        ### Time Series Conclusions

        - The model prediction agrees with the data for the last month of 2022 within the errors.

        - We expect a complex seasonality here, which SARIMA is not able to handle, and is beyond the
        scope of this project.

        - Perhaps looking at data grouped by week would be a better approach to simplify the seasonality.


        """
    )


# ==================================

# ==================================
# Fifth page: Conclusions
if page == pages[4]:

    st.image("assets/traffic.png")
    image_link = f'<a href=https://auto.hindustantimes.com/auto/news/traffic-jam-as-long-as-700-kms-paris-chokes-on-day-one-of-fresh-lockdown-41604116978417.html> Image credit: Bloomberg'
    st.caption(image_link, unsafe_allow_html=True)

    st.write('## Conclusions')

    text = """
    - All four models that we tested performed well in predicting whether a user would be injured or not \
    in an accident. The best performing models were the XGBoost and the DNN, with a recall in the \
    category of interest of 83% and 90%, respectively.

    - We decided not to include the rest of the data in our model training, as the data quality from before \
    2019 is subpar (many missing values), and the information on the security equipment is missing. \
    Computational cost would be prohibitively high for the scope of this project and with our available \
    computing resources.

    - The most important feature in predicting whether or not a user was injured was if they are wearing \
    a seatbelt. In fact, two of the top 5 features were safety equipment.

    - As the percentage of injuries and fatalities remains constant, the best way to save lives is to \
    reduce the overall number of accidents.
    """

    st.markdown(text)

    st.write("## Outlook")
    st.write("**Include data from before 2019**")
    st.write("- This would require more computing resources than we had available for this project.")
    st.write("- This would also require a substantial time investment in additional feature engineering, for instance to find a way to unify the safety equipment variables from before and after 2019.")
    st.write("**Exploring two alternative approaches: severity by accident and severity by vehicle.**")
    st.write("- We're considering a new measure of severity, exploring cumulative, mean, or maximum severity by accident. These potential refinements hold the promise of enhancing predictive accuracy and relevance.")
    st.write("**4-Class Classification model**")
    st.write("- A first attempt at this gave abysmal results, but it would be interesting to follow up.")

# ==================================
