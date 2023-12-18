import streamlit as st
import datetime
import statsmodels.api as sm, numpy as np, pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

def section4():
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
