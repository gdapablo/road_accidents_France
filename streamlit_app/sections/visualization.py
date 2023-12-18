import streamlit as st
from matplotlib.pyplot import imread
import streamlit.components.v1 as components

def section2():
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
