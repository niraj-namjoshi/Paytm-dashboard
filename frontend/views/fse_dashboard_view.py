"""
FSE Team Dashboard View
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from services.api_client import api_client

def show_fse_dashboard(dashboard_data):
    """Display FSE team dashboard with real analysis data"""
    user_info = dashboard_data["user_info"]
    department = user_info['department']
    
    # Map department to team endpoint
    team_mapping = {
        "UX Team": "ux",
        "Payment Team": "payments", 
        "Dev Team": "dev"
    }
    
    team_endpoint = team_mapping.get(department, "ux")
    
    # Header
    st.title(f"{department} Dashboard")
    st.markdown(f"**Real-time Analysis for {department}**")
    st.markdown("---")
    
    # Get team-specific data from backend
    team_data = api_client.get_team_data(team_endpoint, st.session_state.token)
    
    if not team_data:
        st.error("❌ Failed to load team analysis data")
        return
    
    data = team_data.get("data", {})
    global_context = team_data.get("global_context", {})
    
    # Issue Clusters - Move to top
    clusters = data.get("clusters", [])
    st.subheader("Critical Issues Analysis")
    
    if clusters:
        # Sort clusters by severity (high -> medium -> low)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_clusters = sorted(clusters, key=lambda x: severity_order.get(x.get("severity", "low").lower(), 3))
        
        # Count severity levels
        severity_counts = {"low": 0, "medium": 0, "high": 0}
        for cluster in clusters:
            severity = cluster.get("severity", "low").lower()
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        # Display severity breakdown
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Low Severity", severity_counts["low"])
        with col2:
            st.metric("Medium Severity", severity_counts["medium"])
        with col3:
            st.metric("High Severity", severity_counts["high"])
        
        st.markdown("---")
        
        # Display each cluster with pills like in the image
        for i, cluster in enumerate(sorted_clusters, 1):
            severity = cluster.get("severity", "low").lower()
            cluster_summary = cluster.get('cluster_summary', 'No summary available')
            cluster_count = cluster.get('cluster_count', 0)
            
            # Create a container for the issue
            with st.container():
                # Create the pills inline with the text
                if severity == "high":
                    severity_color = "#dc3545"  # Red
                    severity_text = "HIGH"
                elif severity == "medium":
                    severity_color = "#fd7e14"  # Orange
                    severity_text = "MEDIUM"
                else:
                    severity_color = "#28a745"  # Green
                    severity_text = "LOW"
                
                # Display issue with inline pills
                st.markdown(f'''
                <div style="margin: 8px 0; padding: 12px; background-color: rgba(255,255,255,0.02); border-radius: 8px; border-left: 3px solid {severity_color};">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span style="background-color: {severity_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold;">{severity_text}</span>
                        <span style="border: 1px solid rgba(255,255,255,0.4); color: rgba(255,255,255,0.8); padding: 2px 8px; border-radius: 12px; font-size: 11px; background-color: rgba(255,255,255,0.1);">{cluster_count} reviews</span>
                    </div>
                    <div style="color: #ffffff; font-size: 14px; line-height: 1.4;">{cluster_summary}</div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Show sample comments in expander
                with st.expander("View Customer Comments", expanded=False):
                    comments = cluster.get("comments", [])
                    if comments:
                        for j, comment in enumerate(comments[:3], 1):
                            rating = comment.get("rating", "N/A")
                            location = comment.get("location", "Unknown")
                            st.write(f"{j}. *\"{comment.get('comment', 'No comment')}\"* - {rating}⭐ ({location})")
    else:
        st.info("No issues found for your team! Great job!")
    
    st.markdown("---")
    
    # Team Overview Metrics - Clean and professional
    st.subheader("Team Analysis Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'''
        <div style="text-align: center; padding: 20px; background-color: rgba(255,255,255,0.05); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin: 8px 0;">
            <h2 style="color: #ffffff; margin: 0; font-size: 32px; font-weight: 600;">{len(clusters)}</h2>
            <p style="color: rgba(255,255,255,0.7); margin: 8px 0 0 0; font-size: 14px; font-weight: 500;">Issue Clusters</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        nps_data = data.get("team_nps_breakdown", {})
        nps_score = nps_data.get('nps', 0)
        nps_color = "#28a745" if nps_score > 0 else "#dc3545" if nps_score < -20 else "#ffc107"
        st.markdown(f'''
        <div style="text-align: center; padding: 20px; background-color: rgba(255,255,255,0.05); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin: 8px 0;">
            <h2 style="color: {nps_color}; margin: 0; font-size: 32px; font-weight: 600;">{nps_score}</h2>
            <p style="color: rgba(255,255,255,0.7); margin: 8px 0 0 0; font-size: 14px; font-weight: 500;">Team NPS</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # NPS Breakdown - Clean and minimal
    st.subheader("Customer Satisfaction (NPS)")
    if nps_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            promoters = nps_data.get("promoters", 0)
            st.markdown(f'''
            <div style="text-align: center; padding: 16px; background-color: rgba(255,255,255,0.03); border-radius: 6px; border: 1px solid rgba(40, 167, 69, 0.3);">
                <h3 style="color: #28a745; margin: 0; font-size: 24px; font-weight: 600;">{promoters}</h3>
                <p style="color: rgba(255,255,255,0.6); margin: 6px 0 0 0; font-size: 13px;">Promoters</p>
            </div>
            ''', unsafe_allow_html=True)
        with col2:
            neutrals = nps_data.get("neutrals", 0)
            st.markdown(f'''
            <div style="text-align: center; padding: 16px; background-color: rgba(255,255,255,0.03); border-radius: 6px; border: 1px solid rgba(255, 193, 7, 0.3);">
                <h3 style="color: #ffc107; margin: 0; font-size: 24px; font-weight: 600;">{neutrals}</h3>
                <p style="color: rgba(255,255,255,0.6); margin: 6px 0 0 0; font-size: 13px;">Neutrals</p>
            </div>
            ''', unsafe_allow_html=True)
        with col3:
            detractors = nps_data.get("detractors", 0)
            st.markdown(f'''
            <div style="text-align: center; padding: 16px; background-color: rgba(255,255,255,0.03); border-radius: 6px; border: 1px solid rgba(220, 53, 69, 0.3);">
                <h3 style="color: #dc3545; margin: 0; font-size: 24px; font-weight: 600;">{detractors}</h3>
                <p style="color: rgba(255,255,255,0.6); margin: 6px 0 0 0; font-size: 13px;">Detractors</p>
            </div>
            ''', unsafe_allow_html=True)
        with col4:
            nps_score = nps_data.get("nps", 0)
            nps_color = "#28a745" if nps_score > 0 else "#dc3545" if nps_score < -20 else "#ffc107"
            st.markdown(f'''
            <div style="text-align: center; padding: 16px; background-color: rgba(255,255,255,0.03); border-radius: 6px; border: 1px solid rgba(255,255,255,0.2);">
                <h3 style="color: {nps_color}; margin: 0; font-size: 24px; font-weight: 600;">{nps_score}</h3>
                <p style="color: rgba(255,255,255,0.6); margin: 6px 0 0 0; font-size: 13px;">NPS Score</p>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sentiment Analysis - Plotly Donut Chart
    st.subheader("Sentiment Distribution")
    sentiment_data = data.get("team_sentiment_distribution", {})
    if sentiment_data:
        # Prepare data for donut chart
        labels = ['Positive', 'Neutral', 'Negative']
        values = [sentiment_data.get('positive', 0), sentiment_data.get('neutral', 0), sentiment_data.get('negative', 0)]
        colors = ['#00CC96', '#FFA15A', '#EF553B']
        
        
        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent+value',
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Team Sentiment Distribution",
            showlegend=True,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    
    # Global Context
    st.markdown("---")
    st.subheader("Global Context")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Total Reviews Analyzed:** {global_context.get('total_reviews', 0)}")
    
    with col2:
        global_sentiment = global_context.get('sentiment_distribution', {})
        total_negative = global_sentiment.get('negative', 0)
        st.write(f"**Global Negative Reviews:** {total_negative}")
    
    # Debug API Output
    st.markdown("---")
    with st.expander("Debug: Backend API Output"):
        st.subheader("Complete API Response")
        st.json(team_data)
    
    # Action Items
    st.markdown("---")
    st.subheader("Recommended Actions")
    
    if clusters:
        high_severity = [c for c in clusters if c.get("severity", "").lower() == "high"]
        if high_severity:
            st.error(f"**Urgent:** {len(high_severity)} high-severity issues require immediate attention!")
        
        medium_severity = [c for c in clusters if c.get("severity", "").lower() == "medium"]
        if medium_severity:
            st.warning(f"**Important:** {len(medium_severity)} medium-severity issues need resolution.")
        
        if nps_data.get("nps", 0) < -20:
            st.error("**Critical:** Team NPS is significantly negative. Immediate action required!")
        elif nps_data.get("nps", 0) < 0:
            st.warning("**Attention:** Team NPS is negative. Focus on customer satisfaction improvements.")
    else:
        st.success("**Excellent:** No critical issues detected for your team!")