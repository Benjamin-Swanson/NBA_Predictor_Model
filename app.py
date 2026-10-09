"""NBA Predictor site. Run:  streamlit run app.py      (opens http://localhost:8501)

Pages only read precomputed files (data/*.json, models/model.joblib); nothing here calls nba_api.
"""
import streamlit as st

st.set_page_config(page_title="NBA Predictor", page_icon="🏀", layout="wide")
st.navigation([
    st.Page("pages/1_Games.py", title="Games", icon="🏀", default=True),
    st.Page("pages/2_Season_Outlook.py", title="Season outlook", icon="🏆"),
]).run()
