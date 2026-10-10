"""The Hardwood Model site. Run:  streamlit run app.py      (opens http://localhost:8501)

Pages only read precomputed files (data/*.json, models/model.joblib); nothing here calls nba_api.
Look and feel: .streamlit/config.toml (theme) + ui.py (CSS, logos, cards).
"""
import streamlit as st

import ui

st.set_page_config(page_title="The Hardwood Model", page_icon=str(ui.BRAND / "favicon.png"), layout="wide")
ui.style()
pg = st.navigation([
    st.Page("views/1_Games.py", title="Games", default=True),
    st.Page("views/2_Season_Outlook.py", title="Season outlook", url_path="season"),
], position="top")
ui.brand()
pg.run()
