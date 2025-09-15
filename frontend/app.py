"""
Main Streamlit Frontend Application
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pages.login import show_login_page
from pages.dashboard import show_dashboard_page

# Configure Streamlit page
st.set_page_config(
    page_title="Login System",
    page_icon="🔐",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    section[data-testid="stSidebar"] {display: none !important;}
    .css-1d391kg {display: none}
    .css-1rs6os {display: none}
    .css-17eq0hr {display: none}
    .css-1lcbmhc {display: none}
    .css-1outpf7 {display: none}
    header[data-testid="stHeader"] {display: none !important;}
</style>
""", unsafe_allow_html=True)

def main():
    """Main application logic"""
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_data" not in st.session_state:
        st.session_state.user_data = None
    if "token" not in st.session_state:
        st.session_state.token = None

    # Route to appropriate page
    if st.session_state.logged_in:
        show_dashboard_page()
    else:
        show_login_page()

if __name__ == "__main__":
    main()
