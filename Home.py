import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
import os
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import seaborn as sns

if 'df' not in st.session_state:
        st.session_state["df"] = None

def load_sidebar():
    # with st.sidebar:
    st.write("")

def Home():
    load_sidebar()
    st.divider()
    tabs = st.tabs(["Home"])
    with tabs[0]:
        st.header("Welcome to Edupreneurialship Prediction Model")
        st.divider()
    
    
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        st.session_state["df"] = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.write(st.session_state["df"])
        option = st.selectbox("Select an option:", ["Show dataset dimensions", "Display data description", "Verify data integrity", "Visualization"]) #"Summarize numerical data statistics", "Summarize categorical data"])

        if option == "Show dataset dimensions":
            st.write("Dataset dimensions:", st.session_state["df"].shape)
        elif option == "Display data description":
            st.write("Data description:", st.session_state["df"].describe())
        elif option == "Verify data integrity":
            missing_values = st.session_state["df"].isnull().sum()
            # Convert the Series to a DataFrame for better display
            missing_values_df = pd.DataFrame(missing_values, columns=["Missing Values"])  # Give a column name

            st.write("Missing values in each column:")
            st.table(missing_values_df)
            # st.write("Missing values in each column:", st.session_state["df"].isnull().sum())
        
        # elif option == "Summarize numerical data statistics":
        #     st.write("Numerical data statistics:", st.session_state["df"].describe(include=[np.number]))
        # elif option == "Summarize categorical data":
        #     st.write("Categorical data summary:", st.session_state["df"].describe(include=[np.object_]))
        
        elif option == "Visualization":
        # Visualization options
            visualization_type = st.selectbox("Select a visualization type:", ["Bar Chart"])

            if visualization_type == "Bar Chart":
                
                column = st.selectbox("Select a column for bar chart:", st.session_state["df"].columns)
                fig, ax = plt.subplots(figsize=(10, 4))
                counts = st.session_state["df"][column].value_counts()  # Get value counts

                # Plot the bar chart
                bars = counts.plot(kind='bar', ax=ax)

                # Add labels on top of bars
                for bar in bars.patches:  # Iterate over the bar objects
                    yval = bar.get_height()  # Get the height of the bar
                    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, str(int(yval)), ha='center', va='bottom') # Add text label

                st.pyplot(fig)  # Display the plot