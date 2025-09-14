"""
FastAPI Backend for Login System with LLM Analysis
"""

import json
import csv
import os
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import jwt
import sys
import threading
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.services.auth_service import AuthService
from api.models.user import User, LoginRequest, LoginResponse, DashboardData
from api.models.review import ReviewsAnalysisRequest
from api.LLMs.llmgenerator import analyze_reviews_from_json, get_embeddings, cluster_embeddings, top_representatives, generate_cluster_summary, calculate_nps, generate_team_tags_for_reviews
from api.LLMs import sentiment_analyzer

# JWT Configuration
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(title="Login System API", version="1.0.0")

# CORS middleware to allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Login System API is running"}

@app.post("/api/auth/login", response_model=LoginResponse)
async def login(login_request: LoginRequest):
    """Authenticate user and return JWT token"""
    if not AuthService.validate_credentials(login_request.username, login_request.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    user_data = AuthService.get_user_data(login_request.username)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": login_request.username}, expires_delta=access_token_expires
    )
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user_data=user_data
    )

@app.get("/api/auth/me", response_model=User)
async def get_current_user(username: str = Depends(verify_token)):
    """Get current authenticated user data"""
    user_data = AuthService.get_user_data(username)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return User(
        username=username,
        user_type=user_data["type"],
        department=user_data.get("department"),
        location=user_data.get("location")
    )

@app.get("/api/dashboard/data", response_model=DashboardData)
async def get_dashboard_data(username: str = Depends(verify_token)):
    """Get dashboard data for authenticated user"""
    user_data = AuthService.get_user_data(username)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Generate dashboard data based on user type
    if user_data["type"] == "FSE Team":
        return DashboardData(
            metrics={
                "active_projects": 12,
                "team_members": 8,
                "completed_tasks": 45
            },
            features=AuthService.get_department_features(user_data["department"]),
            user_info={
                "type": user_data["type"],
                "department": user_data["department"]
            }
        )
    elif user_data["type"] == "Area Manager":
        return DashboardData(
            metrics={
                "total_teams": 15,
                "active_projects": 32,
                "team_members": 120,
                "monthly_revenue": "₹2.5M"
            },
            features=AuthService.get_manager_features(),
            user_info={
                "type": user_data["type"],
                "location": user_data["location"]
            }
        )

# Global variables to store analysis results and incremental data
analysis_cache = {}
incremental_cache = {
    "last_processed_date": None,
    "last_processed_id": 0,
    "full_refresh_time": None
}
refresh_lock = threading.Lock()

def load_reviews_from_csv(since_date=None, since_id=None):
    """Load reviews from the CSV file with optional filtering"""
    try:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "reviews.csv")
        reviews = []
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                review = {
                    "id": int(row["id"]),
                    "comment": row["comment"],
                    "rating": int(row["rating"]),
                    "location": row["location"],
                    "date": row["date"]
                }
                
                # Filter for incremental loading - ID-based filtering is primary
                if since_id and review["id"] <= since_id:
                    print(f"🔍 Skipping review ID {review['id']} (since_id: {since_id})")
                    continue
                
                # Only use date filtering as backup if ID filtering fails
                if since_date and not since_id and review["date"] <= since_date:
                    print(f"🔍 Skipping review date {review['date']} (since_date: {since_date})")
                    continue
                
                print(f"✅ Including new review ID {review['id']}: {review['comment'][:50]}...")
                reviews.append(review)
        
        print(f"📊 Total new reviews found: {len(reviews)}")
        return reviews
    except Exception as e:
        print(f"Error loading reviews: {e}")
        return []

def load_new_reviews_only():
    """Load only new reviews since last processing - prioritize ID-based filtering"""
    print(f"🔍 Loading new reviews with filters:")
    print(f"   - since_id: {incremental_cache['last_processed_id']}")
    print(f"   - since_date: {incremental_cache['last_processed_date']}")
    
    # Use ID-based filtering as primary method
    return load_reviews_from_csv(
        since_id=incremental_cache["last_processed_id"],
        since_date=None  # Don't use date filtering when ID is available
    )

def auto_analyze_reviews_on_startup():
    """Automatically analyze reviews when server starts"""
    try:
        print("🔄 Loading and analyzing reviews automatically...")
        reviews_data = load_reviews_from_csv()
        
        if reviews_data:
            print(f"📊 Found {len(reviews_data)} reviews. Starting full analysis...")
            analysis_results = analyze_reviews_from_json(reviews_data)
            analysis_cache["latest"] = analysis_results
            
            # Initialize incremental cache
            with refresh_lock:
                incremental_cache["last_processed_date"] = max(r["date"] for r in reviews_data)
                incremental_cache["last_processed_id"] = max(r["id"] for r in reviews_data)
                incremental_cache["full_refresh_time"] = datetime.now()
                print(f"🔧 Initialized incremental cache:")
                print(f"   - last_processed_id: {incremental_cache['last_processed_id']}")
                print(f"   - last_processed_date: {incremental_cache['last_processed_date']}")
            
            print("✅ Reviews analyzed successfully and cached!")
            print(f"📈 Analysis summary:")
            print(f"   - Total reviews: {analysis_results.get('total_reviews', 0)}")
            print(f"   - Global NPS: {analysis_results.get('global_nps_breakdown', {}).get('nps', 0)}")
            print(f"   - Teams with issues: {list(analysis_results.get('problem_clusters_by_team', {}).keys())}")
        else:
            print("⚠️ No reviews found in CSV file")
    except Exception as e:
        print(f"❌ Error during auto-analysis: {e}")

def incremental_analyze_new_reviews():
    """Perform incremental analysis on new reviews with proper clustering and NPS updates"""
    try:
        print(f"🔄 Debug: Current incremental cache state:")
        print(f"   - last_processed_id: {incremental_cache['last_processed_id']}")
        print(f"   - last_processed_date: {incremental_cache['last_processed_date']}")
        
        with refresh_lock:
            new_reviews = load_new_reviews_only()
            
        if not new_reviews:
            print("🔄 No new reviews found for incremental analysis")
            return False
            
        print(f"🔄 Found {len(new_reviews)} new reviews for incremental analysis...")
        
        # Get sentiment for new reviews
        new_comments = [r['comment'] for r in new_reviews]
        sentiment_results = sentiment_analyzer.get_sentiment_scores(new_comments)
        
        # Add sentiment to new reviews
        for i, review in enumerate(new_reviews):
            sentiment_res = sentiment_results[i]
            review['sentiment'] = ['negative', 'neutral', 'positive'][sentiment_res['sentiment']]
            review['sentiment_scores'] = sentiment_res['scores']
        
        # Generate team tags for ALL new reviews
        review_team_tags = generate_team_tags_for_reviews(new_reviews)
        
        # Assign team tags to reviews
        for review in new_reviews:
            comment = review.get('comment', '')
            review['team_tag'] = review_team_tags.get(comment, 'Other')
        
        # Filter negative/neutral reviews for clustering
        negative_new_reviews = [
            r for r in new_reviews if r['sentiment'] in ['negative', 'neutral']
        ]
        
        # Process negative reviews for clustering
        new_clusters_by_team = {"UX": [], "Dev": [], "Payments": [], "Other": []}
        
        if negative_new_reviews:
            # Group negative reviews by team
            team_negative_reviews = {"UX": [], "Dev": [], "Payments": [], "Other": []}
            for review in negative_new_reviews:
                team = review.get('team_tag', 'Other')
                if team in team_negative_reviews:
                    team_negative_reviews[team].append(review)
            
            # Create micro-clusters for each team's new negative reviews
            for team, team_reviews in team_negative_reviews.items():
                if team_reviews:
                    # For small batches, create simple clusters based on similarity
                    team_comments = [r['comment'] for r in team_reviews]
                    
                    if len(team_comments) >= 2:
                        # Use embeddings and clustering for multiple reviews
                        team_embeddings = get_embeddings(team_comments)
                        team_labels, team_centroids = cluster_embeddings(team_embeddings, max_clusters=min(3, len(team_comments)))
                        team_reps = top_representatives(team_comments, team_embeddings, team_labels, team_centroids, top_n=3)
                        
                        # Generate cluster summaries for this team's new issues
                        cluster_sizes = {}
                        clusters_map = {}
                        for i, label in enumerate(team_labels):
                            if label not in clusters_map:
                                clusters_map[label] = []
                                cluster_sizes[label] = 0
                            clusters_map[label].append(team_reviews[i])
                            cluster_sizes[label] += 1
                        
                        generated_summaries = generate_cluster_summary(team_reps, cluster_sizes)
                        
                        # Create new clusters for this team
                        for summary in generated_summaries:
                            cluster_id = summary.get('cluster_id', 0)
                            if isinstance(cluster_id, str):
                                cluster_id = int(cluster_id.split(',')[0].strip()) if cluster_id.split(',')[0].strip().isdigit() else 0
                            
                            cluster_data = {
                                "cluster_summary": summary.get('cluster_summary', 'New issue identified'),
                                "severity": summary.get('severity', 'medium'),
                                "comments": clusters_map.get(cluster_id, team_reviews),
                                "cluster_count": len(clusters_map.get(cluster_id, team_reviews))
                            }
                            new_clusters_by_team[team].append(cluster_data)
                    else:
                        # Single review - create simple cluster
                        cluster_data = {
                            "cluster_summary": f"New {team.lower()} issue: {team_comments[0][:50]}...",
                            "severity": "low" if len(team_reviews) == 1 else "medium",
                            "comments": team_reviews,
                            "cluster_count": len(team_reviews)
                        }
                        new_clusters_by_team[team].append(cluster_data)
        
        # Update analysis cache with comprehensive changes
        with refresh_lock:
            if "latest" in analysis_cache:
                current_analysis = analysis_cache["latest"]
                
                # Update total reviews count
                current_analysis["total_reviews"] += len(new_reviews)
                
                # Update global sentiment distribution
                for review in new_reviews:
                    sentiment = review['sentiment']
                    if sentiment in current_analysis["sentiment_distribution"]:
                        current_analysis["sentiment_distribution"][sentiment] += 1
                
                # Collect all ratings by team for NPS recalculation
                team_ratings = {"UX": [], "Dev": [], "Payments": [], "Other": []}
                
                # Add existing ratings from current analysis (approximate from NPS data)
                for team, nps_data in current_analysis.get("team_nps_breakdown", {}).items():
                    if isinstance(nps_data, dict):
                        # Reconstruct approximate ratings from NPS breakdown
                        promoters = nps_data.get('promoters', 0)
                        neutrals = nps_data.get('neutrals', 0)
                        detractors = nps_data.get('detractors', 0)
                        
                        # Add approximate ratings (this is a simplification)
                        team_ratings[team].extend([5] * promoters)  # Promoters as 5-star
                        team_ratings[team].extend([3] * neutrals)   # Neutrals as 3-star
                        team_ratings[team].extend([1] * detractors) # Detractors as 1-star
                
                # Add new review ratings
                for review in new_reviews:
                    team = review.get('team_tag', 'Other')
                    rating = review.get('rating')
                    if team in team_ratings and rating:
                        team_ratings[team].append(rating)
                
                # Recalculate NPS for each team
                for team, ratings in team_ratings.items():
                    if ratings:
                        new_nps = calculate_nps(ratings)
                        current_analysis["team_nps_breakdown"][team] = new_nps
                
                # Update team-level sentiment distribution
                for review in new_reviews:
                    team = review.get('team_tag', 'Other')
                    sentiment = review['sentiment']
                    
                    if team not in current_analysis.get("team_sentiment_distribution", {}):
                        current_analysis["team_sentiment_distribution"][team] = {"positive": 0, "neutral": 0, "negative": 0}
                    
                    if sentiment in current_analysis["team_sentiment_distribution"][team]:
                        current_analysis["team_sentiment_distribution"][team][sentiment] += 1
                
                # Add new clusters to team data
                for team, new_clusters in new_clusters_by_team.items():
                    if new_clusters and team in current_analysis.get("problem_clusters_by_team", {}):
                        current_analysis["problem_clusters_by_team"][team]["clusters"].extend(new_clusters)
                        
                        # Update team count
                        team_count = sum(cluster["cluster_count"] for cluster in current_analysis["problem_clusters_by_team"][team]["clusters"])
                        current_analysis["problem_clusters_by_team"][team]["team_count"] = team_count
                        
                        # Update team-level NPS and sentiment in team data
                        current_analysis["problem_clusters_by_team"][team]["team_nps_breakdown"] = current_analysis["team_nps_breakdown"].get(team, {})
                        current_analysis["problem_clusters_by_team"][team]["team_sentiment_distribution"] = current_analysis["team_sentiment_distribution"].get(team, {})
                
                # Update location-level metrics
                location_ratings = {}
                for review in new_reviews:
                    location = review.get('location')
                    if location:
                        sentiment = review['sentiment']
                        rating = review.get('rating')
                        
                        # Update location sentiment
                        if location not in current_analysis.get("location_sentiment_distribution", {}):
                            current_analysis["location_sentiment_distribution"][location] = {"positive": 0, "neutral": 0, "negative": 0}
                        
                        if sentiment in current_analysis["location_sentiment_distribution"][location]:
                            current_analysis["location_sentiment_distribution"][location][sentiment] += 1
                        
                        # Collect ratings for location NPS update
                        if rating:
                            if location not in location_ratings:
                                location_ratings[location] = []
                            location_ratings[location].append(rating)
                
                # Update location NPS (simplified - would need full rating history in production)
                for location, new_ratings in location_ratings.items():
                    if location in current_analysis.get("location_nps_breakdown", {}):
                        # For simplicity, just add the new ratings impact
                        # In production, you'd maintain full rating history
                        existing_nps = current_analysis["location_nps_breakdown"][location]
                        if isinstance(existing_nps, dict):
                            # Approximate update - in production, maintain full rating lists
                            additional_nps = calculate_nps(new_ratings)
                            # Simple weighted average (this is approximate)
                            total_existing = existing_nps.get('promoters', 0) + existing_nps.get('neutrals', 0) + existing_nps.get('detractors', 0)
                            if total_existing > 0:
                                weight_existing = total_existing / (total_existing + len(new_ratings))
                                weight_new = len(new_ratings) / (total_existing + len(new_ratings))
                                
                                updated_nps = {
                                    'nps': round(existing_nps.get('nps', 0) * weight_existing + additional_nps.get('nps', 0) * weight_new, 2),
                                    'promoters': existing_nps.get('promoters', 0) + additional_nps.get('promoters', 0),
                                    'neutrals': existing_nps.get('neutrals', 0) + additional_nps.get('neutrals', 0),
                                    'detractors': existing_nps.get('detractors', 0) + additional_nps.get('detractors', 0)
                                }
                                current_analysis["location_nps_breakdown"][location] = updated_nps
                
                # Add new positive reviews
                positive_new_reviews = [r for r in new_reviews if r['sentiment'] == 'positive']
                if positive_new_reviews:
                    current_analysis["positive_reviews"].extend(positive_new_reviews)
                
                # Update global NPS with new reviews
                all_new_ratings = [r.get('rating') for r in new_reviews if r.get('rating')]
                if all_new_ratings:
                    # Get existing global NPS data
                    existing_global_nps = current_analysis.get("global_nps_breakdown", {})
                    if existing_global_nps:
                        # Approximate global NPS update
                        existing_total = existing_global_nps.get('promoters', 0) + existing_global_nps.get('neutrals', 0) + existing_global_nps.get('detractors', 0)
                        new_nps_data = calculate_nps(all_new_ratings)
                        
                        if existing_total > 0:
                            weight_existing = existing_total / (existing_total + len(all_new_ratings))
                            weight_new = len(all_new_ratings) / (existing_total + len(all_new_ratings))
                            
                            updated_global_nps = {
                                'nps': round(existing_global_nps.get('nps', 0) * weight_existing + new_nps_data.get('nps', 0) * weight_new, 2),
                                'promoters': existing_global_nps.get('promoters', 0) + new_nps_data.get('promoters', 0),
                                'neutrals': existing_global_nps.get('neutrals', 0) + new_nps_data.get('neutrals', 0),
                                'detractors': existing_global_nps.get('detractors', 0) + new_nps_data.get('detractors', 0)
                            }
                            current_analysis["global_nps_breakdown"] = updated_global_nps
                
                # Update tracking info
                incremental_cache["last_processed_date"] = max(r["date"] for r in new_reviews)
                incremental_cache["last_processed_id"] = max(r["id"] for r in new_reviews)
                
                print(f"✅ Comprehensive incremental analysis completed for {len(new_reviews)} new reviews")
                print(f"   - New clusters created: {sum(len(clusters) for clusters in new_clusters_by_team.values())}")
                print(f"   - Teams updated: {[team for team, clusters in new_clusters_by_team.items() if clusters]}")
                return True
        
        return False
    except Exception as e:
        print(f"❌ Error during incremental analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def should_do_full_refresh():
    """Check if it's time for a full refresh (every 30 minutes)"""
    if not incremental_cache["full_refresh_time"]:
        return True
    
    time_since_full = datetime.now() - incremental_cache["full_refresh_time"]
    return time_since_full.total_seconds() > 1800  # 30 minutes

def background_refresh_task():
    """Background task that runs every 2 minutes"""
    while True:
        try:
            time.sleep(120)  # Wait 2 minutes
            
            if should_do_full_refresh():
                print("🔄 Performing full refresh (30-minute cycle)...")
                auto_analyze_reviews_on_startup()
            else:
                print("🔄 Performing incremental refresh (2-minute cycle)...")
                incremental_analyze_new_reviews()
                
        except Exception as e:
            print(f"❌ Error in background refresh task: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

@app.on_event("startup")
async def startup_event():
    """Run analysis when server starts and start background refresh"""
    auto_analyze_reviews_on_startup()
    
    # Start background refresh task
    refresh_thread = threading.Thread(target=background_refresh_task, daemon=True)
    refresh_thread.start()
    print("🚀 Background refresh task started (2-minute incremental, 30-minute full)")

@app.get("/api/reviews/load")
async def load_and_analyze_reviews(username: str = Depends(verify_token)):
    """Load reviews from CSV file and analyze them (force full refresh)"""
    try:
        # Load reviews from CSV file
        reviews_data = load_reviews_from_csv()
        
        if not reviews_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No reviews found in the CSV file"
            )
        
        # Perform full analysis
        analysis_results = analyze_reviews_from_json(reviews_data)
        
        # Cache results and update incremental tracking
        with refresh_lock:
            analysis_cache["latest"] = analysis_results
            incremental_cache["last_processed_date"] = max(r["date"] for r in reviews_data)
            incremental_cache["last_processed_id"] = max(r["id"] for r in reviews_data)
            incremental_cache["full_refresh_time"] = datetime.now()
        
        return {
            "message": f"Successfully analyzed {len(reviews_data)} reviews (full refresh)",
            "analysis_results": analysis_results
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/api/reviews/refresh")
async def trigger_incremental_refresh(username: str = Depends(verify_token)):
    """Manually trigger incremental refresh"""
    try:
        success = incremental_analyze_new_reviews()
        if success:
            return {
                "message": "Incremental refresh completed successfully",
                "last_processed_id": incremental_cache["last_processed_id"],
                "last_processed_date": incremental_cache["last_processed_date"]
            }
        else:
            return {
                "message": "No new reviews found for incremental refresh",
                "last_processed_id": incremental_cache["last_processed_id"],
                "last_processed_date": incremental_cache["last_processed_date"]
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Incremental refresh failed: {str(e)}"
        )

@app.get("/api/refresh/status")
async def get_refresh_status(username: str = Depends(verify_token)):
    """Get current refresh status and timing information"""
    return {
        "last_processed_id": incremental_cache["last_processed_id"],
        "last_processed_date": incremental_cache["last_processed_date"],
        "last_full_refresh": incremental_cache["full_refresh_time"].isoformat() if incremental_cache["full_refresh_time"] else None,
        "next_full_refresh_in_seconds": 1800 - (datetime.now() - incremental_cache["full_refresh_time"]).total_seconds() if incremental_cache["full_refresh_time"] else 0,
        "background_refresh_active": True,
        "refresh_interval_seconds": 120,
        "full_refresh_interval_seconds": 1800
    }

@app.post("/api/analyze/reviews")
async def analyze_reviews(request: ReviewsAnalysisRequest, username: str = Depends(verify_token)):
    """Analyze reviews using LLM and store results (full analysis)"""
    try:
        # Convert Pydantic models to dict format expected by llmgenerator
        reviews_data = [review.dict() for review in request.reviews]
        
        # Perform full analysis
        analysis_results = analyze_reviews_from_json(reviews_data)
        
        # Cache results and update tracking
        with refresh_lock:
            analysis_cache["latest"] = analysis_results
            if reviews_data:
                incremental_cache["last_processed_date"] = max(r.get("date", "") for r in reviews_data)
                incremental_cache["last_processed_id"] = max(r.get("id", 0) for r in reviews_data)
                incremental_cache["full_refresh_time"] = datetime.now()
        
        return analysis_results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/api/reviews/raw")
async def get_raw_reviews(username: str = Depends(verify_token)):
    """Get raw reviews from CSV file"""
    try:
        reviews_data = load_reviews_from_csv()
        return {
            "total_reviews": len(reviews_data),
            "reviews": reviews_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load reviews: {str(e)}"
        )

@app.get("/api/teams/ux")
async def get_ux_data(username: str = Depends(verify_token)):
    """Get UX team specific data with complete sentiment and NPS"""
    if "latest" not in analysis_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis data available. Please run analysis first."
        )
    
    results = analysis_cache["latest"]
    ux_problem_data = results.get("problem_clusters_by_team", {}).get("UX", {})
    
    # Get complete team data including all sentiment and NPS
    team_data = {
        "team_count": ux_problem_data.get("team_count", 0),
        "clusters": ux_problem_data.get("clusters", []),
        "team_sentiment_distribution": results.get("team_sentiment_distribution", {}).get("UX", {}),
        "team_nps_breakdown": results.get("team_nps_breakdown", {}).get("UX", {})
    }
    
    return {
        "team": "UX",
        "data": team_data,
        "global_context": {
            "total_reviews": results.get("total_reviews", 0),
            "sentiment_distribution": results.get("sentiment_distribution", {})
        }
    }

@app.get("/api/teams/payments")
async def get_payments_data(username: str = Depends(verify_token)):
    """Get Payments team specific data with complete sentiment and NPS"""
    if "latest" not in analysis_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis data available. Please run analysis first."
        )
    
    results = analysis_cache["latest"]
    payments_problem_data = results.get("problem_clusters_by_team", {}).get("Payments", {})
    
    # Get complete team data including all sentiment and NPS
    team_data = {
        "team_count": payments_problem_data.get("team_count", 0),
        "clusters": payments_problem_data.get("clusters", []),
        "team_sentiment_distribution": results.get("team_sentiment_distribution", {}).get("Payments", {}),
        "team_nps_breakdown": results.get("team_nps_breakdown", {}).get("Payments", {})
    }
    
    return {
        "team": "Payments",
        "data": team_data,
        "global_context": {
            "total_reviews": results.get("total_reviews", 0),
            "sentiment_distribution": results.get("sentiment_distribution", {})
        }
    }

@app.get("/api/teams/dev")
async def get_dev_data(username: str = Depends(verify_token)):
    """Get Dev team specific data with complete sentiment and NPS"""
    if "latest" not in analysis_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis data available. Please run analysis first."
        )
    
    results = analysis_cache["latest"]
    dev_problem_data = results.get("problem_clusters_by_team", {}).get("Dev", {})
    
    # Get complete team data including all sentiment and NPS
    team_data = {
        "team_count": dev_problem_data.get("team_count", 0),
        "clusters": dev_problem_data.get("clusters", []),
        "team_sentiment_distribution": results.get("team_sentiment_distribution", {}).get("Dev", {}),
        "team_nps_breakdown": results.get("team_nps_breakdown", {}).get("Dev", {})
    }
    
    return {
        "team": "Dev",
        "data": team_data,
        "global_context": {
            "total_reviews": results.get("total_reviews", 0),
            "sentiment_distribution": results.get("sentiment_distribution", {})
        }
    }

@app.get("/api/sentiment/location")
async def get_location_sentiment(username: str = Depends(verify_token)):
    """Get location-based sentiment distribution"""
    if "latest" not in analysis_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis data available. Please run analysis first."
        )
    
    results = analysis_cache["latest"]
    
    return {
        "location_sentiment_distribution": results.get("location_sentiment_distribution", {}),
        "global_sentiment": results.get("sentiment_distribution", {}),
        "total_reviews": results.get("total_reviews", 0)
    }

@app.get("/api/nps/location")
async def get_location_nps(username: str = Depends(verify_token)):
    """Get location/area-wise NPS breakdown"""
    if "latest" not in analysis_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis data available. Please run analysis first."
        )
    
    results = analysis_cache["latest"]
    
    return {
        "location_nps_breakdown": results.get("location_nps_breakdown", {}),
        "global_nps": results.get("global_nps_breakdown", {}),
        "total_reviews": results.get("total_reviews", 0)
    }

@app.get("/api/nps/scores")
async def get_nps_scores(username: str = Depends(verify_token)):
    """Get NPS scores for all teams and global"""
    if "latest" not in analysis_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis data available. Please run analysis first."
        )
    
    results = analysis_cache["latest"]
    teams_data = results.get("problem_clusters_by_team", {})
    
    nps_data = {
        "global_nps": results.get("global_nps_breakdown", {}),
        "team_nps": {}
    }
    
    for team_name, team_info in teams_data.items():
        if team_info and "team_nps_breakdown" in team_info:
            nps_data["team_nps"][team_name] = team_info["team_nps_breakdown"]
    
    return nps_data

@app.get("/api/locations/{location}")
async def get_location_data(location: str, username: str = Depends(verify_token)):
    """Get location-specific data with complete sentiment and NPS"""
    if "latest" not in analysis_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis data available. Please run analysis first."
        )
    
    results = analysis_cache["latest"]
    location_problem_data = results.get("problem_clusters_by_location", {}).get(location, {})
    
    # Get complete location data including all sentiment and NPS
    location_data = {
        "location_count": location_problem_data.get("location_count", 0),
        "clusters": location_problem_data.get("clusters", []),
        "location_sentiment_distribution": results.get("location_sentiment_distribution", {}).get(location, {}),
        "location_nps_breakdown": results.get("location_nps_breakdown", {}).get(location, {})
    }
    
    return {
        "location": location,
        "data": location_data,
        "global_context": {
            "total_reviews": results.get("total_reviews", 0),
            "sentiment_distribution": results.get("sentiment_distribution", {})
        }
    }

@app.get("/api/analysis/summary")
async def get_analysis_summary(username: str = Depends(verify_token)):
    """Get complete analysis summary"""
    if "latest" not in analysis_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis data available. Please run analysis first."
        )
    
    return analysis_cache["latest"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
