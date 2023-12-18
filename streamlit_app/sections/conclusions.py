import streamlit as st

def section5():
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
