#!/usr/bin/env python3
import streamlit as st, config
import pandas as pd

from sections import introduction, visualization, modelling, time, conclusions

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
    introduction.section1()
# ==================================

# ==================================
# Second page: Pre-processing and Visualization
if page == pages[1]:
    visualization.section2()
# ==================================

# ==================================
# Third page: Modelling
if page == pages[2]:
    modelling.section3()
# ==================================

# ==================================
# Fourth page: Time Series
if page == pages[3]:
    time.section4()
# ==================================

# ==================================
# Fifth page: Conclusions
if page == pages[4]:
    conclusions.section5()
# ==================================
