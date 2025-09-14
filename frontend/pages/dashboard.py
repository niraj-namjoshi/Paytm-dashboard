"""
Dashboard page for authenticated users
"""

import streamlit as st
from services.api_client import api_client
from views.fse_dashboard_view import show_fse_dashboard
from views.manager_dashboard_view import show_manager_dashboard

def show_dashboard_page():
    """Display dashboard based on user type"""
    # Verify user is logged in
    if not st.session_state.get("logged_in", False):
        st.error("Please login first")
        st.session_state.logged_in = False
        st.rerun()
        return
    
    # Get dashboard data from API
    dashboard_data = api_client.get_dashboard_data(st.session_state.token)
    
    if not dashboard_data:
        st.error("Failed to load dashboard data")
        return
    
    # Display logout button in top-right corner
    st.markdown("""
    <style>
    .logout-button {
        position: fixed;
        top: 10px;
        right: 20px;
        z-index: 999;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([10, 1])
        with col2:
            if st.button("Logout", key="logout_btn", help="Sign out of your account"):
                # Clear session state
                st.session_state.logged_in = False
                st.session_state.token = None
                st.session_state.user_data = None
                st.rerun()
    
    # Route to appropriate dashboard view
    user_type = dashboard_data["user_info"]["type"]
    
    if user_type == "FSE Team":
        show_fse_dashboard(dashboard_data)
    elif user_type == "Area Manager":
        show_manager_dashboard(dashboard_data)
    else:
        st.error("Unknown user type")
