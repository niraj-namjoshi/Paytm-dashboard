"""
Login page for Streamlit frontend
"""

import streamlit as st
from services.api_client import api_client

def show_login_page():
    """Display login page"""
    st.title("🔐 Login System")
    st.markdown("---")
    
    # Check backend connectivity
    if not api_client.health_check():
        st.error("⚠️ Backend API is not accessible. Please ensure the backend server is running on http://localhost:8000")
        st.stop()
    
    # Display demo credentials
    st.info("**Demo Credentials for Testing:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**FSE Teams:**")
        st.code("""Username: ux_team
Password: password123
Department: UX Team""")
        
        st.code("""Username: payment_team
Password: password123
Department: Payment Team""")
        
        st.code("""Username: dev_team
Password: password123
Department: Dev Team""")
    
    with col2:
        st.markdown("**Area Managers:**")
        st.code("""Username: mumbai_manager
Password: password123
Location: Mumbai""")
        
        st.code("""Username: bangalore_manager
Password: password123
Location: Bangalore""")
    
    st.markdown("---")
    
    # Login form
    with st.form("login_form"):
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if username and password:
                # Attempt login
                login_response = api_client.login(username, password)
                
                if login_response:
                    # Store authentication data in session
                    st.session_state.logged_in = True
                    st.session_state.token = login_response["access_token"]
                    st.session_state.user_data = login_response["user_data"]
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
            else:
                st.warning("⚠️ Please enter both username and password")
