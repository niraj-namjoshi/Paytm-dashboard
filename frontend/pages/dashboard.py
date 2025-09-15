"""
Dashboard page for authenticated users
"""

import streamlit as st
import time
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
    
    # Initialize sync settings
    if "sync_with_backend" not in st.session_state:
        st.session_state.sync_with_backend = True
    if "check_interval" not in st.session_state:
        st.session_state.check_interval = 2  # Auto-trigger every 2 seconds
    if "last_check" not in st.session_state:
        st.session_state.last_check = time.time()
    if "last_data_version" not in st.session_state:
        st.session_state.last_data_version = None
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    # Sync logic - check backend for data updates
    current_time = time.time()
    if (st.session_state.sync_with_backend and 
        current_time - st.session_state.last_check >= st.session_state.check_interval):
        
        # 1) Auto-trigger incremental analysis of new reviews
        _ = api_client.trigger_incremental_refresh(st.session_state.token)
        
        # 2) Check backend data version after trigger
        version_data = api_client.get_data_version(st.session_state.token)
        if version_data:
            current_version = version_data.get("version")
            if (st.session_state.last_data_version is not None and 
                current_version != st.session_state.last_data_version):
                # Data has been updated on backend, refresh frontend
                st.session_state.last_data_version = current_version
                st.session_state.last_refresh = current_time
                st.session_state.last_check = current_time
                st.rerun()
            elif st.session_state.last_data_version is None:
                # First time, just store the version
                st.session_state.last_data_version = current_version
        
        st.session_state.last_check = current_time
    
    # Get dashboard data from API
    dashboard_data = api_client.get_dashboard_data(st.session_state.token)
    
    if not dashboard_data:
        st.error("Failed to load dashboard data")
        return
    
    # Display controls in top section
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
        col1, col2, col3, col4 = st.columns([6, 2, 1, 1])
        
        with col1:
            if st.button("🔁 Analyze New Reviews", key="analyze_now_btn", help="Trigger backend incremental analysis of new reviews"):
                resp = api_client.trigger_incremental_refresh(st.session_state.token)
                if resp:
                    st.success(resp.get("message", "Incremental refresh triggered"))
                    st.session_state.last_refresh = time.time()
                    st.rerun()
                else:
                    st.warning("No new reviews found or refresh failed")
        
        with col2:
            # Backend sync toggle
            sync_enabled = st.toggle("Sync with Backend", value=st.session_state.sync_with_backend, key="sync_toggle")
            st.session_state.sync_with_backend = sync_enabled
        
        with col3:
            # Check interval selector and manual refresh
            if st.session_state.sync_with_backend:
                interval = st.selectbox("Check Interval (sec)", [2, 5, 10, 15], 
                                      index=[2, 5, 10, 15].index(st.session_state.check_interval),
                                      key="check_interval_select")
                st.session_state.check_interval = interval
            else:
                if st.button("🔄 Refresh Now", key="manual_refresh_btn", help="Manually refresh data"):
                    st.session_state.last_refresh = time.time()
                    st.rerun()
        
        with col4:
            if st.button("Logout", key="logout_btn", help="Sign out of your account"):
                # Clear session state
                st.session_state.logged_in = False
                st.session_state.token = None
                st.session_state.user_data = None
                st.rerun()
    
    # Show sync status
    if st.session_state.sync_with_backend:
        time_since_check = int(current_time - st.session_state.last_check)
        next_check = st.session_state.check_interval - time_since_check
        if next_check > 0:
            st.info(f"🔄 Auto-analyzing every {st.session_state.check_interval}s. Next check in {next_check} seconds")
        else:
            st.info("🔄 Triggering analysis and checking for backend updates...")
        
        # Show current data version if available
        if st.session_state.last_data_version is not None:
            st.caption(f"Data version: {st.session_state.last_data_version}")
        
        # Show backend refresh status summary
        status = api_client.get_refresh_status(st.session_state.token)
        if status:
            st.caption(
                f"Last processed ID: {status.get('last_processed_id')} | "
                f"Last processed date: {status.get('last_processed_date')}"
            )
    
    # Add last updated timestamp
    st.caption(f"Last updated: {time.strftime('%H:%M:%S', time.localtime(st.session_state.last_refresh))}")
    
    # Route to appropriate dashboard view
    user_type = dashboard_data["user_info"]["type"]
    
    if user_type == "FSE Team":
        show_fse_dashboard(dashboard_data)
    elif user_type == "Area Manager":
        show_manager_dashboard(dashboard_data)
    else:
        st.error("Unknown user type")
