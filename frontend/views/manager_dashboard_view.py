"""
Area Manager Dashboard View
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from services.api_client import api_client

def show_manager_dashboard(dashboard_data):
    """Display area manager dashboard with real location-based analysis data"""
    user_info = dashboard_data["user_info"]
    location = user_info['location']
    
    # Header
    st.title(f"{location} Manager Dashboard")
    st.markdown(f"**Real-time Analysis for {location} Region**")
    st.markdown("---")
    
    # Get location-specific data from backend using correct endpoints
    location_data = api_client.get_location_data(location, st.session_state.token)
    location_nps_data = api_client.get_location_nps(st.session_state.token)
    location_sentiment_data = api_client.get_location_sentiment(st.session_state.token)
    team_nps_data = api_client.get_nps_scores(st.session_state.token)
    critical_issues_data = api_client.get_location_critical_issues(location, st.session_state.token)
    
    if not location_data:
        st.error("Failed to load location analysis data")
        return
    
    data = location_data.get("data", {})
    
    # Combine data from correct endpoints
    global_context = {
        "total_reviews": location_nps_data.get("total_reviews", 0) if location_nps_data else 0,
        "sentiment_distribution": location_sentiment_data.get("global_sentiment", {}) if location_sentiment_data else {},
        "global_nps_breakdown": location_nps_data.get("global_nps", {}) if location_nps_data else {},
        "team_nps_breakdown": team_nps_data.get("team_nps", {}) if team_nps_data else {}
    }
    
    # Update data with correct location-specific info
    if location_nps_data:
        data["location_nps_breakdown"] = location_nps_data.get("location_nps_breakdown", {}).get(location, {})
    
    if location_sentiment_data:
        data["location_sentiment_distribution"] = location_sentiment_data.get("location_sentiment_distribution", {}).get(location, {})
    
    
    # Regional Overview Metrics - Color coded like the reference image
    st.subheader("Regional Analysis Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        regional_reviews = sum(data.get("location_sentiment_distribution", {}).values()) if data.get("location_sentiment_distribution") else 0
        st.markdown(f'''
        <div style="text-align: center; padding: 20px; background-color: rgba(255,255,255,0.05); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin: 8px 0;">
            <h2 style="color: #ffffff; margin: 0; font-size: 32px; font-weight: 600;">{regional_reviews}</h2>
            <p style="color: rgba(255,255,255,0.7); margin: 8px 0 0 0; font-size: 14px; font-weight: 500;">Regional Reviews</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        nps_data = data.get("location_nps_breakdown", {})
        nps_score = nps_data.get('nps', 0)
        # Color coding based on NPS score
        if nps_score >= 0:
            nps_color = "#28a745"  # Green for positive
        elif nps_score >= -20:
            nps_color = "#ffc107"  # Yellow for neutral/slightly negative
        else:
            nps_color = "#dc3545"  # Red for very negative
        
        st.markdown(f'''
        <div style="text-align: center; padding: 20px; background-color: rgba(255,255,255,0.05); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin: 8px 0;">
            <h2 style="color: {nps_color}; margin: 0; font-size: 32px; font-weight: 600;">{nps_score}</h2>
            <p style="color: rgba(255,255,255,0.7); margin: 8px 0 0 0; font-size: 14px; font-weight: 500;">Regional NPS</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # NPS Breakdown - Color coded like the reference image
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
            # Color coding based on NPS score
            if nps_score >= 0:
                nps_color = "#28a745"  # Green for positive
            elif nps_score >= -20:
                nps_color = "#ffc107"  # Yellow for neutral/slightly negative
            else:
                nps_color = "#dc3545"  # Red for very negative
            
            st.markdown(f'''
            <div style="text-align: center; padding: 16px; background-color: rgba(255,255,255,0.03); border-radius: 6px; border: 1px solid rgba(255,255,255,0.2);">
                <h3 style="color: {nps_color}; margin: 0; font-size: 24px; font-weight: 600;">{nps_score}</h3>
                <p style="color: rgba(255,255,255,0.6); margin: 6px 0 0 0; font-size: 13px;">NPS Score</p>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sentiment Analysis - Plotly Donut Chart
    st.subheader("Regional Sentiment Distribution")
    sentiment_data = data.get("location_sentiment_distribution", {})
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
            title="Regional Sentiment Distribution",
            showlegend=True,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Critical Issues Section - Location-based
    st.subheader(f"🚨 Critical Issues in {location}")
    if critical_issues_data:
        if critical_issues_data and critical_issues_data.get("critical_issues"):
            critical_issues = critical_issues_data["critical_issues"]
            
            st.metric(
                label="High Severity Issues",
                value=len(critical_issues),
                delta=None
            )
            
            # Single centered display for high severity issues only
            st.markdown(f'''
            <div style="text-align: center; padding: 20px; background-color: rgba(220, 53, 69, 0.1); border-radius: 8px; border: 1px solid rgba(220, 53, 69, 0.3); margin: 8px 0;">
                <h2 style="color: #dc3545; margin: 0; font-size: 32px; font-weight: 600;">{len(critical_issues)}</h2>
                <p style="color: rgba(255,255,255,0.7); margin: 8px 0 0 0; font-size: 14px; font-weight: 500;">High Severity Issues</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Display critical issues
            st.markdown("### High Severity Issues Details")
            for i, issue in enumerate(critical_issues[:5]):  # Show top 5 high severity issues
                severity = issue.get("severity", "high")
                count = issue.get("cluster_count", 0)
                summary = issue.get("cluster_summary", "No summary available")
                
                # Color coding for high severity (all issues are high severity now)
                severity_color = "#dc3545"
                bg_color = "rgba(220, 53, 69, 0.1)"
                border_color = "rgba(220, 53, 69, 0.3)"
                
                st.markdown(f'''
                <div style="padding: 15px; background-color: {bg_color}; border-radius: 8px; border: 1px solid {border_color}; margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="color: {severity_color}; font-weight: 600; text-transform: uppercase; font-size: 12px;">
                            {severity} SEVERITY
                        </span>
                        <span style="color: rgba(255,255,255,0.8); font-size: 14px; font-weight: 500;">
                            {count} reports
                        </span>
                    </div>
                    <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 14px; line-height: 1.4;">
                        {summary}
                    </p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Show sample comments for critical issues
                if severity == "high" and st.expander(f"View sample comments for issue #{i+1}"):
                    comments = issue.get("comments", [])[:3]  # Show first 3 comments
                    for comment in comments:
                        comment_text = comment.get("comment", "No comment text")
                        rating = comment.get("rating", "N/A")
                        st.markdown(f"**Rating: {rating}/5** - {comment_text}")
        else:
            st.success(f"✅ No critical issues found in {location}. Great work!")
    else:
        st.warning("Unable to load critical issues data")
    
    # Global vs Regional Comparison
    st.markdown("---")
    st.subheader("Global vs Regional Comparison")
    
    # Global Sentiment vs Regional Sentiment
    st.subheader("Sentiment Analysis Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Global Sentiment Distribution**")
        global_sentiment = global_context.get('sentiment_distribution', {})
        if global_sentiment:
            # Global sentiment metrics
            global_pos = global_sentiment.get('positive', 0)
            global_neu = global_sentiment.get('neutral', 0) 
            global_neg = global_sentiment.get('negative', 0)
            global_total = global_pos + global_neu + global_neg
            
            if global_total > 0:
                st.metric("Positive", f"{global_pos} ({global_pos/global_total*100:.1f}%)")
                st.metric("Neutral", f"{global_neu} ({global_neu/global_total*100:.1f}%)")
                st.metric("Negative", f"{global_neg} ({global_neg/global_total*100:.1f}%)")
            else:
                st.write("No global sentiment data available")
    
    with col2:
        st.write(f"**{location} Regional Sentiment**")
        if sentiment_data:
            # Regional sentiment metrics
            regional_pos = sentiment_data.get('positive', 0)
            regional_neu = sentiment_data.get('neutral', 0)
            regional_neg = sentiment_data.get('negative', 0)
            regional_total = regional_pos + regional_neu + regional_neg
            
            if regional_total > 0:
                st.metric("Positive", f"{regional_pos} ({regional_pos/regional_total*100:.1f}%)")
                st.metric("Neutral", f"{regional_neu} ({regional_neu/regional_total*100:.1f}%)")
                st.metric("Negative", f"{regional_neg} ({regional_neg/regional_total*100:.1f}%)")
            else:
                st.write("No regional sentiment data available")
    
    # Global NPS vs Regional NPS
    st.subheader("NPS Score Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Global NPS Score**")
        global_nps = global_context.get('global_nps_breakdown', {})
        if global_nps:
            global_nps_score = global_nps.get('nps', 0)
            color = "normal" if global_nps_score > 0 else "inverse"
            st.metric("Global NPS", f"{global_nps_score}", delta_color=color)
            
            # Global NPS breakdown
            st.write("**Global NPS Breakdown:**")
            st.write(f"Promoters: {global_nps.get('promoters', 0)}")
            st.write(f"Neutrals: {global_nps.get('neutrals', 0)}")
            st.write(f"Detractors: {global_nps.get('detractors', 0)}")
    
    with col2:
        st.write(f"**{location} Regional NPS**")
        if nps_data:
            regional_nps_score = nps_data.get('nps', 0)
            color = "normal" if regional_nps_score > 0 else "inverse"
            st.metric("Regional NPS", f"{regional_nps_score}", delta_color=color)
            
            # Regional NPS breakdown
            st.write("**Regional NPS Breakdown:**")
            st.write(f"Promoters: {nps_data.get('promoters', 0)}")
            st.write(f"Neutrals: {nps_data.get('neutrals', 0)}")
            st.write(f"Detractors: {nps_data.get('detractors', 0)}")
    
    # Team NPS Scores - Color coded like the reference image
    st.subheader("Team Performance (NPS by Team)")
    team_nps_data = global_context.get('team_nps_breakdown', {})
    if team_nps_data:
        col1, col2, col3, col4 = st.columns(4)
        
        teams = ['UX', 'Dev', 'Payments', 'Other']
        cols = [col1, col2, col3, col4]
        
        for team, col in zip(teams, cols):
            with col:
                team_nps = team_nps_data.get(team, {})
                if team_nps:
                    team_score = team_nps.get('nps', 0)
                    # Color coding based on team NPS score
                    if team_score >= 0:
                        team_color = "#28a745"  # Green for positive
                    elif team_score >= -20:
                        team_color = "#ffc107"  # Yellow for neutral/slightly negative
                    else:
                        team_color = "#dc3545"  # Red for very negative
                    
                    st.markdown(f'''
                    <div style="text-align: center; padding: 16px; background-color: rgba(255,255,255,0.03); border-radius: 6px; border: 1px solid rgba(255,255,255,0.1); margin: 4px 0;">
                        <h3 style="color: {team_color}; margin: 0; font-size: 20px; font-weight: 600;">{team_score}</h3>
                        <p style="color: rgba(255,255,255,0.6); margin: 6px 0 0 0; font-size: 12px;">{team} Team</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div style="text-align: center; padding: 16px; background-color: rgba(255,255,255,0.03); border-radius: 6px; border: 1px solid rgba(255,255,255,0.1); margin: 4px 0;">
                        <h3 style="color: rgba(255,255,255,0.4); margin: 0; font-size: 20px; font-weight: 600;">N/A</h3>
                        <p style="color: rgba(255,255,255,0.6); margin: 6px 0 0 0; font-size: 12px;">{team} Team</p>
                    </div>
                    ''', unsafe_allow_html=True)
    
    # Summary Stats
    st.write("**Summary Statistics**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Total Reviews Analyzed:** {global_context.get('total_reviews', 0)}")
    with col2:
        regional_reviews = sum(sentiment_data.values()) if sentiment_data else 0
        st.write(f"**Regional Reviews:** {regional_reviews}")
    
    # Debug API Output
    st.markdown("---")
    with st.expander("Debug: Backend API Output"):
        st.subheader("Complete API Response")
        st.json(location_data)
    
    # Management Actions
    st.markdown("---")
    st.subheader("Regional Management Actions")
    
    nps_score = nps_data.get("nps", 0) if nps_data else 0
    if nps_score < -20:
        st.error("**Critical:** Regional NPS is significantly negative. Immediate action required!")
    elif nps_score < 0:
        st.warning("**Attention:** Regional NPS is negative. Focus on customer satisfaction improvements.")
    else:
        st.success("**Excellent:** Regional performance is positive!")
    
    # Quick Management Tools
    st.markdown("---")
    st.subheader("Management Tools")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Team Analysis"):
            st.success("Loading team performance analysis...")
    
    with col2:
        if st.button("Detailed Reports"):
            st.success("Generating detailed regional reports...")
    
    with col3:
        if st.button("Escalate Issues"):
            st.success("Issue escalation system opened...")
    
    with col4:
        if st.button("Schedule Review"):
            st.success("Team review scheduler opened...")