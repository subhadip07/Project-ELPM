# app.py
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import json
import os
import subprocess

# Import custom functions
from Home import Home
from cluster import classify
from association import association
from sentiment_analysis import *


# Set page configuration
st.set_page_config(
    page_title="Edupreneurialship Prediction Model",
    page_icon=":rocket:",
    layout="wide"
)

# Initialize session state for the DataFrame
if 'df' not in st.session_state:
    st.session_state.df = None

# Initialize session state for login/signup
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None

USER_DATA_FILE = "user_data.json"

def load_user_data():
    try:
        with open(USER_DATA_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"users": {}}
    except json.JSONDecodeError:
        return {"users": {}}

def save_user_data(data):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def check_credentials(username, password, role):
    user_data = load_user_data()
    if username in user_data["users"] and user_data["users"][username]["password"] == password and user_data["users"][username]["role"] == role:
        return True
    else:
        return False

def register_user(username, password, role):
    user_data = load_user_data()
    if username in user_data["users"]:
        return False  # Username already exists
    user_data["users"][username] = {"password": password, "role": role}
    save_user_data(user_data)
    return True

# --- LOGIN/SIGNUP ---
if not st.session_state.logged_in:
    if not st.session_state.show_signup:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["student", "faculty"])
        if st.button("Login"):
            if check_credentials(username, password, role):
                st.session_state.logged_in = True
                st.session_state.user_role = role
                st.success(f"Logged in successfully as {role}!")
                st.rerun()
            else:
                st.error("Invalid username, password, or role.")
        if st.button("Sign Up"):
            st.session_state.show_signup = True
            st.rerun()
    else:
        st.title("Sign Up")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        new_role = st.selectbox("Role", ["student", "faculty"])
        if st.button("Register"):
            if register_user(new_username, new_password, new_role):
                st.success("User registered successfully! Please login.")
                st.session_state.show_signup = False
                st.rerun()
            else:
                st.error("Username already exists.")
        if st.button("Back to Login"):
            st.session_state.show_signup = False
            st.rerun()
else:
    # --- MAIN APP CONTENT ---
    st.title("Edupreneurialship Prediction Model")

    menu_options = ['Home', 'Classification and Prediction', 'Sentiment']
    if st.session_state.user_role == "faculty":
        menu_options.append('Association')

    selected = option_menu(
        'Main Menu',
        menu_options,
        icons=['house', 'list-check', 'graph-up', 'shuffle'] if st.session_state.user_role == "faculty" else ['house', 'list-check', 'graph-up'],
        default_index=0,
        menu_icon="cast",
        orientation="horizontal"
    )
    if selected == "Home":
        Home()
    elif selected == "Classification and Prediction":
        classify()
    elif selected == "Association" and st.session_state.user_role == "faculty":
        association()
    
    elif selected == "Sentiment":
        analyze_sentiment_page()
        

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.rerun()