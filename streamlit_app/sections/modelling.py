import streamlit as st
import matplotlib.pyplot as plt
import glob, pandas as pd, tensorflow as tf

from matplotlib.pyplot import imread
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, precision_recall_curve, classification_report, recall_score
from sklearn.feature_selection import VarianceThreshold
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

def section3():
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
