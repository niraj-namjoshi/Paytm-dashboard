import numpy as np
import re
import math
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from dotenv import load_dotenv
import os
import json
from google import genai
from . import sentiment_analyzer  # Import the sentiment module

# Load environment variables for the API key
load_dotenv()
api_k = os.getenv("api_k")
client = genai.Client(api_key=api_k)

# --- Global Models (loaded once) ---
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    sentiment_analyzer.load_sentiment_model()
    print("✅ All models loaded successfully.")
except Exception as e:
    print(f"❌ Error during model loading: {e}")
    exit()

# --- Core Analysis Functions (remaining in this file) ---

def clean_text(text: str) -> str:
    """Removes HTML tags and normalizes whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_embeddings(texts: List[str]) -> np.ndarray:
    """Generates sentence embeddings for a list of texts."""
    if not texts:
        return np.array([])
    return embedding_model.encode(texts, show_progress_bar=False)

def find_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 20) -> int:
    """
    Finds a suitable number of clusters, prioritizing more clusters for larger datasets
    to ensure better granularity, while still respecting the underlying data structure.
    """
    n_samples = embeddings.shape[0]
    min_clusters = max(2, min(3, n_samples // 8))
    max_clusters = min(max_clusters, n_samples // 2)
    if max_clusters <= min_clusters or n_samples < 2:
        return min_clusters
    
    range_clusters = range(min_clusters, max_clusters + 1)
    inertias = []
    for k in range_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)
    
    try:
        kneedle = KneeLocator(range_clusters, inertias, curve='convex', direction='decreasing')
        elbow_optimal = kneedle.elbow
    except:
        elbow_optimal = None
    
    # Use a heuristic to prioritize more clusters for more granular results
    heuristic_k = max(min_clusters, round(n_samples / 2))

    if elbow_optimal is not None:
        # If elbow is found, take an average to balance optimal clustering with desired granularity
        optimal_k = round((elbow_optimal + heuristic_k) / 2)
    else:
        optimal_k = heuristic_k
    
    optimal_k = max(min_clusters, min(optimal_k, max_clusters))
    return int(optimal_k)

def cluster_embeddings(embeddings: np.ndarray, max_clusters: int = 20) -> tuple:
    """Clusters embeddings using KMeans with an optimized number of clusters."""
    n_samples = embeddings.shape[0]
    
    if n_samples <= 1:
        return np.zeros(n_samples, dtype=int), np.array([np.mean(embeddings, axis=0)])
    
    # Find optimal number of clusters
    n_clusters = find_optimal_clusters(embeddings, max_clusters)
    
    # Ensure n_clusters is valid
    n_clusters = min(n_clusters, n_samples)
    n_clusters = max(1, n_clusters)
    
    # Perform clustering with optimized number
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    return labels, kmeans.cluster_centers_

def top_representatives(texts: List[str], embeddings: np.ndarray, labels: np.ndarray, 
                         centroids: np.ndarray, top_n: int = 6) -> Dict:
    """Finds the most representative reviews for each cluster."""
    reps = {}
    unique_labels = np.unique(labels)
    for cluster_id in unique_labels:
        idxs = np.where(labels == cluster_id)[0]
        cluster_embs = embeddings[idxs]
        centroid = centroids[cluster_id]
        sims = cluster_embs.dot(centroid) / (
            np.linalg.norm(cluster_embs, axis=1) * np.linalg.norm(centroid)
        )
        top_idxs_in_cluster = np.argsort(sims)[-min(top_n, len(idxs)):]
        top_original_idxs = idxs[top_idxs_in_cluster]
        reps[int(cluster_id)] = [texts[i] for i in top_original_idxs]
    return reps

def generate_cluster_summary(reps_by_cluster: Dict[int, List[str]], cluster_sizes: Dict[int, int]) -> List[Dict[str, Any]]:
    """
    Generates a structured list of problem statements for each cluster using an LLM.
    """
    if not reps_by_cluster:
        return []

    # Prepare all clusters data for single LLM call
    all_clusters_data = []
    for cluster_id, reps in reps_by_cluster.items():
        cluster_info = {
            "cluster_id": cluster_id,
            "size": cluster_sizes.get(cluster_id, 0),
            "representative_reviews": reps[:3]
        }
        all_clusters_data.append(cluster_info)
    
    prompt = (
        "Given the following customer reviews, generate one point for each cluster summarizing their main issue: "
        "The severity should be 'low', 'medium', or 'high' based on review count. "
        "The team tag should be a single word chosen ONLY from: 'UX', 'Dev', 'Payments'. "
        "Please provide the output as a JSON array of objects, where each object contains "
        "'cluster_id', 'cluster_summary', 'team_tag', and 'severity'.\n\n"
        "All clusters data:\n"
    )

    for cluster_info in all_clusters_data:
        prompt += f"\nCluster ID: {cluster_info['cluster_id']} (Size: {cluster_info['size']}):"
        for rep in cluster_info['representative_reviews']:
            prompt += f"\n- {rep}"
    
    prompt += "\n\nJSON Output:"

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        raw_output = response.text.strip()
        cleaned_output = raw_output.strip('```json').strip('```').strip()
        json_output = json.loads(cleaned_output)
        
        return json_output
    except Exception as e:
        print(f"Error generating or parsing JSON from LLM: {e}")
        return []

def merge_similar_clusters(clusters: List[Dict[str, Any]], similarity_threshold: float = 0.75) -> List[Dict[str, Any]]:
    """
    Merges clusters that have similar summaries to avoid duplicate issues.
    """
    if len(clusters) <= 1:
        return clusters
    
    # Get embeddings for cluster summaries
    summaries = [cluster.get('cluster_summary', '') for cluster in clusters]
    summary_embeddings = get_embeddings(summaries)
    
    if len(summary_embeddings) == 0:
        return clusters
    
    # Calculate similarity matrix
    merged_clusters = []
    used_indices = set()
    
    for i, cluster in enumerate(clusters):
        if i in used_indices:
            continue
            
        # Find similar clusters
        similar_indices = [i]
        for j in range(i + 1, len(clusters)):
            if j in used_indices:
                continue
                
            # Calculate cosine similarity
            sim = np.dot(summary_embeddings[i], summary_embeddings[j]) / (
                np.linalg.norm(summary_embeddings[i]) * np.linalg.norm(summary_embeddings[j])
            )
            
            if sim >= similarity_threshold:
                similar_indices.append(j)
                used_indices.add(j)
        
        # Merge similar clusters
        if len(similar_indices) > 1:
            merged_comments = []
            total_count = 0
            severities = []
            team_tags = []
            
            for idx in similar_indices:
                merged_comments.extend(clusters[idx].get('comments', []))
                total_count += clusters[idx].get('cluster_count', 0)
                severities.append(clusters[idx].get('severity', 'low'))
                team_tags.append(clusters[idx].get('team_tag', 'Other'))
            
            # Determine merged severity based on total count
            if total_count >= 16:
                merged_severity = 'high'
            elif total_count >= 6:
                merged_severity = 'medium'
            else:
                merged_severity = 'low'
            
            # Use most common team tag
            most_common_team = max(set(team_tags), key=team_tags.count)
            
            # Create comprehensive summary
            base_summary = clusters[i].get('cluster_summary', '')
            merged_cluster = {
                'cluster_summary': f"{base_summary} (consolidated from {len(similar_indices)} related issues)",
                'severity': merged_severity,
                'team_tag': most_common_team,
                'comments': merged_comments,
                'cluster_count': total_count,
                'cluster_id': clusters[i].get('cluster_id', i)
            }
            merged_clusters.append(merged_cluster)
        else:
            merged_clusters.append(cluster)
        
        used_indices.add(i)
    
    return merged_clusters

def generate_team_tags_for_reviews(reviews: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Generates team tags for individual reviews using LLM.
    Returns a dictionary mapping review comments to team tags.
    """
    if not reviews:
        return {}

    # Process reviews in batches to avoid token limits
    batch_size = 20
    all_team_tags = {}
    
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i + batch_size]
        
        prompt = (
            "For each of the following customer reviews, assign a team tag based on the content. "
            "The team tag should be a single word chosen ONLY from: 'UX', 'Dev', 'Payments', 'Other'. "
            "Consider:\n"
            "- UX: UI/design, user experience, interface, layout, visual issues\n"
            "- Dev: Technical bugs, crashes, performance, loading, functionality\n"
            "- Payments: Payment processing, billing, transactions, charges\n"
            "- Other: General feedback not fitting above categories\n\n"
            "Please provide the output as a JSON array of objects, where each object contains "
            "'review_index' (0-based index in this batch) and 'team_tag'.\n\n"
            "Reviews:\n"
        )

        for idx, review in enumerate(batch):
            comment = review.get('comment', '')[:200]  # Limit comment length
            prompt += f"{idx}. {comment}\n"
        
        prompt += "\nJSON Output:"

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash", contents=prompt
            )
            raw_output = response.text.strip()
            cleaned_output = raw_output.strip('```json').strip('```').strip()
            json_output = json.loads(cleaned_output)
            
            # Map results back to original reviews
            for result in json_output:
                review_idx = result.get('review_index', 0)
                team_tag = result.get('team_tag', 'Other')
                if 0 <= review_idx < len(batch):
                    comment = batch[review_idx].get('comment', '')
                    all_team_tags[comment] = team_tag
                    
        except Exception as e:
            print(f"Error generating team tags for batch {i//batch_size + 1}: {e}")
            # Assign "Other" as default if LLM fails
            for review in batch:
                comment = review.get('comment', '')
                all_team_tags[comment] = "Other"
    
    return all_team_tags

def calculate_nps(ratings: List[int]) -> Dict[str, Any]:
    """
    Calculates the Net Promoter Score (NPS) and promoter/detractor/neutral counts
    from a list of 1-5 star ratings.
       - Promoters: 4 or 5 stars
       - Neutrals: 3 stars
       - Detractors: 1 or 2 stars
    """
    if not ratings:
        return {"nps": 0.0, "promoters": 0, "neutrals": 0, "detractors": 0}
    
    promoters = sum(1 for r in ratings if r in [4, 5])
    neutrals = sum(1 for r in ratings if r == 3)
    detractors = sum(1 for r in ratings if r in [1, 2])
    total = len(ratings)
    
    promoter_percent = (promoters / total) * 100
    detractor_percent = (detractors / total) * 100
    
    nps_score = round(promoter_percent - detractor_percent, 2)
    
    return {
        "nps": nps_score,
        "promoters": promoters,
        "neutrals": neutrals,
        "detractors": detractors
    }

def get_sentiment_distribution(reviews: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Calculates the sentiment distribution (positive, neutral, negative)
    for a given list of reviews.
    """
    sentiments = {
        "positive": 0,
        "neutral": 0,
        "negative": 0
    }
    for review in reviews:
        sentiment = review.get('sentiment')
        if sentiment in sentiments:
            sentiments[sentiment] += 1
    return sentiments

def analyze_reviews_from_json(json_reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Performs sentiment analysis and clustering on reviews from a JSON input.
    Returns a structured JSON output of key findings.
    """
    if not json_reviews:
        return {"error": "No reviews provided"}

    # 1. Process all comments
    raw_comments = [item.get('comment', '') for item in json_reviews if item.get('comment')]
    comment_to_original_data = {
        item['comment']: item for item in json_reviews if item.get('comment')
    }

    if not raw_comments:
        return {"error": "No valid comments found in JSON input"}

    sentiment_results = sentiment_analyzer.get_sentiment_scores(raw_comments)

    # 2. Split into positive and negative/neutral
    all_reviews_with_sentiment = []
    for sentiment_res in sentiment_results:
        original_data = comment_to_original_data.get(sentiment_res['text'], {})
        review_with_sentiment = {
            "id": original_data.get('id'),
            "comment": sentiment_res['text'],
            "rating": original_data.get('rating'),
            "location": original_data.get('location'),
            "sentiment": ['negative', 'neutral', 'positive'][sentiment_res['sentiment']],
            "sentiment_scores": sentiment_res['scores']
        }
        all_reviews_with_sentiment.append(review_with_sentiment)

    negative_reviews_with_sentiment = [
        r for r in all_reviews_with_sentiment if r['sentiment'] in ['negative', 'neutral']
    ]
    positive_reviews_with_sentiment = [
        r for r in all_reviews_with_sentiment if r['sentiment'] == 'positive'
    ]

    # 3. Cluster negative/neutral comments
    problem_clusters_by_team = {
        "UX": [],
        "Dev": [],
        "Payments": [],
        "Other": []
    }
    
    if negative_reviews_with_sentiment:
        neg_comments = [r['comment'] for r in negative_reviews_with_sentiment]
        neg_embs = get_embeddings(neg_comments)
        neg_labels, neg_cents = cluster_embeddings(neg_embs, max_clusters=25)
        neg_reps = top_representatives(neg_comments, neg_embs, neg_labels, neg_cents, top_n=5)
        
        clusters_map = {
            i: [] for i in np.unique(neg_labels)
        }
        for i, label in enumerate(neg_labels):
            clusters_map[label].append(negative_reviews_with_sentiment[i])
        
        cluster_sizes = {
            label: len(comments) for label, comments in clusters_map.items()
        }
        
        # 4. Generate cluster summaries and tags using LLM
        generated_summaries = generate_cluster_summary(neg_reps, cluster_sizes)
        
        # 4.5. Create initial cluster data
        initial_clusters = []
        for summary in generated_summaries:
            # Handle malformed cluster_id values from LLM
            cluster_id_raw = summary.get('cluster_id', 0)
            try:
                if isinstance(cluster_id_raw, str):
                    # Take first number if multiple IDs are provided
                    cluster_id = int(cluster_id_raw.split(',')[0].strip())
                else:
                    cluster_id = int(cluster_id_raw)
            except (ValueError, TypeError):
                cluster_id = 0
            
            team_tag = summary.get('team_tag', 'Other').strip()
            
            cluster_data = {
                "cluster_summary": summary.get('cluster_summary'),
                "severity": summary.get('severity'),
                "team_tag": team_tag,
                "comments": clusters_map.get(cluster_id, []),
                "cluster_count": len(clusters_map.get(cluster_id, [])),
                "cluster_id": cluster_id
            }
            initial_clusters.append(cluster_data)
        
        # 4.6. Merge similar clusters to avoid duplicates
        merged_clusters = merge_similar_clusters(initial_clusters, similarity_threshold=0.7)
        
        # 5. Group merged clusters by team
        for cluster_data in merged_clusters:
            team_tag = cluster_data.get('team_tag', 'Other')
            
            final_cluster = {
                "cluster_summary": cluster_data.get('cluster_summary'),
                "severity": cluster_data.get('severity'),
                "comments": cluster_data.get('comments', []),
                "cluster_count": cluster_data.get('cluster_count', 0)
            }

            if team_tag in problem_clusters_by_team:
                problem_clusters_by_team[team_tag].append(final_cluster)
            else:
                problem_clusters_by_team['Other'].append(final_cluster)
    
    # 6. Add team-level counts, NPS, and sentiment distribution
    final_team_data = {}
    for team, clusters in problem_clusters_by_team.items():
        final_team_data[team] = {
            "team_count": len([comment for cluster in clusters for comment in cluster['comments']]),
            "team_nps_breakdown": calculate_nps([]),
            "team_sentiment_distribution": get_sentiment_distribution([]),
            "clusters": clusters
        }
    
    # 6.1. Calculate complete team-level metrics from ALL reviews using LLM team tags
    team_sentiment_distribution = {}
    team_nps_breakdown = {}
    
    # Initialize team data structures
    for team in ["UX", "Dev", "Payments", "Other"]:
        team_sentiment_distribution[team] = {"positive": 0, "neutral": 0, "negative": 0}
        team_nps_breakdown[team] = []
    
    # Generate LLM-based team tags for all reviews
    print("🤖 Generating team tags using LLM for all reviews...")
    review_team_tags = generate_team_tags_for_reviews(all_reviews_with_sentiment)
    
    # Assign all reviews to teams based on LLM-generated tags and calculate complete metrics
    for review in all_reviews_with_sentiment:
        comment = review.get('comment', '')
        team = review_team_tags.get(comment, "Other")  # Use LLM-generated team tag
        
        # Update sentiment distribution
        sentiment = review.get('sentiment')
        if sentiment in team_sentiment_distribution[team]:
            team_sentiment_distribution[team][sentiment] += 1
        
        # Collect ratings for team-based NPS
        rating = review.get('rating')
        if rating is not None:
            team_nps_breakdown[team].append(rating)
    
    # Calculate NPS for each team
    team_nps_scores = {}
    for team, ratings in team_nps_breakdown.items():
        team_nps_scores[team] = calculate_nps(ratings)
    
    # Update final team data with complete metrics
    for team in final_team_data:
        final_team_data[team]["team_sentiment_distribution"] = team_sentiment_distribution[team]
        final_team_data[team]["team_nps_breakdown"] = team_nps_scores[team]

    # 7. Add location-based sentiment distribution, NPS, and problem clusters
    location_sentiment_distribution = {}
    location_nps_breakdown = {}
    problem_clusters_by_location = {}
    
    # Initialize location data structures from all reviews
    for review in all_reviews_with_sentiment:
        location = review.get('location')
        if location and location not in location_sentiment_distribution:
            location_sentiment_distribution[location] = {"positive": 0, "neutral": 0, "negative": 0}
            location_nps_breakdown[location] = []
            problem_clusters_by_location[location] = {"location_count": 0, "clusters": []}
    
    # Calculate location-based metrics from all reviews
    for review in all_reviews_with_sentiment:
        location = review.get('location')
        if location:
            sentiment = review.get('sentiment')
            if sentiment in location_sentiment_distribution[location]:
                location_sentiment_distribution[location][sentiment] += 1
            
            # Collect ratings for location-based NPS
            rating = review.get('rating')
            if rating is not None:
                location_nps_breakdown[location].append(rating)
    
    # Group negative clusters by location for manager dashboard
    for team_data in final_team_data.values():
        for cluster in team_data.get("clusters", []):
            for comment in cluster.get("comments", []):
                location = comment.get('location')
                if location and location in problem_clusters_by_location:
                    # Check if this cluster already exists for this location
                    existing_cluster = None
                    for loc_cluster in problem_clusters_by_location[location]["clusters"]:
                        if loc_cluster.get("cluster_summary") == cluster.get("cluster_summary"):
                            existing_cluster = loc_cluster
                            break
                    
                    if not existing_cluster:
                        # Create new location cluster
                        location_cluster = {
                            "cluster_summary": cluster.get("cluster_summary"),
                            "severity": cluster.get("severity"),
                            "comments": [c for c in cluster.get("comments", []) if c.get('location') == location],
                            "cluster_count": len([c for c in cluster.get("comments", []) if c.get('location') == location])
                        }
                        if location_cluster["cluster_count"] > 0:
                            problem_clusters_by_location[location]["clusters"].append(location_cluster)
    
    # Update location counts
    for location in problem_clusters_by_location:
        total_location_issues = sum(cluster["cluster_count"] for cluster in problem_clusters_by_location[location]["clusters"])
        problem_clusters_by_location[location]["location_count"] = total_location_issues
    
    # Calculate NPS for each location
    location_nps_scores = {}
    for location, ratings in location_nps_breakdown.items():
        location_nps_scores[location] = calculate_nps(ratings)
    
    # 8. Compile the final result, including global NPS
    all_ratings = [r.get('rating') for r in all_reviews_with_sentiment if r.get('rating') is not None]
    global_nps = calculate_nps(all_ratings)

    sentiment_counts = {
        "positive": len(positive_reviews_with_sentiment),
        "neutral": len([r for r in negative_reviews_with_sentiment if r['sentiment'] == 'neutral']),
        "negative": len([r for r in negative_reviews_with_sentiment if r['sentiment'] == 'negative']),
    }
    
    return {
        "total_reviews": len(json_reviews),
        "sentiment_distribution": sentiment_counts,
        "global_nps_breakdown": global_nps,
        "problem_clusters_by_team": final_team_data,
        "problem_clusters_by_location": problem_clusters_by_location,
        "team_sentiment_distribution": team_sentiment_distribution,
        "team_nps_breakdown": team_nps_scores,
        "location_sentiment_distribution": location_sentiment_distribution,
        "location_nps_breakdown": location_nps_scores,
        "positive_reviews": positive_reviews_with_sentiment
    }