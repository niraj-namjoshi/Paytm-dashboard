# 🔐 Dashboard v3 - Intelligent Review Analysis System

A comprehensive full-stack application that combines user authentication with AI-powered review analysis. The system automatically processes customer reviews, performs sentiment analysis, clusters issues by team responsibility, and provides detailed analytics through both API endpoints and a user-friendly Streamlit interface.

## 🏗️ Architecture

**Separated Backend & Frontend Architecture:**
- **Backend**: FastAPI with JWT authentication + LLM-powered review analysis
- **Frontend**: Streamlit with role-based dashboards
- **Data Processing**: Automated sentiment analysis, clustering, and NPS calculation
- **AI Integration**: Google Gemini for intelligent issue categorization

## 📁 Project Structure

```
dashboard v3/
├── backend/                    # FastAPI Backend
│   ├── api/
│   │   ├── LLMs/              # AI Analysis Modules
│   │   │   ├── llmgenerator.py    # Main analysis engine
│   │   │   ├── sentiment_analyzer.py # Sentiment analysis
│   │   │   └── __init__.py
│   │   ├── models/            # Pydantic Models
│   │   │   ├── user.py           # User & auth models
│   │   │   ├── review.py         # Review analysis models
│   │   │   └── __init__.py
│   │   ├── services/          # Business Logic
│   │   │   ├── auth_service.py   # Authentication service
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── main.py                # FastAPI application
│   ├── requirements.txt       # Backend dependencies
│   └── .env                   # Environment variables
├── frontend/                  # Streamlit Frontend
│   ├── pages/                 # Page Components
│   │   ├── login.py              # Login interface
│   │   ├── dashboard.py          # Main dashboard
│   │   └── __init__.py
│   ├── services/              # API Communication
│   │   ├── api_client.py         # Backend API client
│   │   └── __init__.py
│   ├── views/                 # Dashboard Views
│   │   ├── fse_dashboard_view.py # FSE team dashboard
│   │   ├── manager_dashboard_view.py # Manager dashboard
│   │   └── __init__.py
│   ├── app.py                 # Main Streamlit app
│   └── requirements.txt       # Frontend dependencies
├── Data/
│   └── reviews.json           # Customer reviews dataset
└── README.md                  # This file
```

## 🚀 Features

### 🔐 Authentication System
- **Hierarchical User Access**: FSE Teams & Area Managers
- **JWT Token Authentication**: Secure API access
- **Role-based Dashboards**: Different views per user type
- **Session Management**: Secure login/logout functionality

### 🤖 AI-Powered Review Analysis
- **Automatic Processing**: Reviews analyzed on server startup
- **Sentiment Analysis**: RoBERTa-based sentiment classification
- **Intelligent Clustering**: Groups similar issues automatically
- **Team Assignment**: AI categorizes issues (UX/Dev/Payments)
- **NPS Calculation**: Global and location-based Net Promoter Score

### 📊 Analytics & Insights
- **Team-specific Dashboards**: Targeted issue views
- **Location Analytics**: City-wise sentiment and NPS
- **Problem Clustering**: Grouped similar customer complaints
- **Performance Metrics**: Comprehensive KPI tracking
- **Real-time Data**: Live analysis results

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Google Gemini API key

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   # Create .env file with your Google API key
   echo "api_k=YOUR_GOOGLE_GEMINI_API_KEY" > .env
   ```

4. **Start the backend server:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Streamlit app:**
   ```bash
   streamlit run app.py --server.port 8501
   ```

## 👥 Demo Credentials

### FSE Teams
- **Username**: `ux_team` | **Password**: `password123` | **Department**: UX Team
- **Username**: `payment_team` | **Password**: `password123` | **Department**: Payment Team  
- **Username**: `dev_team` | **Password**: `password123` | **Department**: Dev Team

### Area Managers
- **Username**: `mumbai_manager` | **Password**: `password123` | **Location**: Mumbai
- **Username**: `bangalore_manager` | **Password**: `password123` | **Location**: Bangalore

## 🔗 API Endpoints

### Authentication
- `GET /` - Health check
- `POST /api/auth/login` - User authentication
- `GET /api/auth/me` - Current user info
- `GET /api/dashboard/data` - Dashboard data

### Review Analysis
- `GET /api/reviews/load` - Auto-load & analyze reviews
- `POST /api/analyze/reviews` - Analyze custom reviews
- `GET /api/reviews/raw` - Raw review data

### Team Analytics
- `GET /api/teams/ux` - UX team issues & metrics
- `GET /api/teams/payments` - Payment team data
- `GET /api/teams/dev` - Development team insights

### Performance Metrics
- `GET /api/nps/scores` - Team-wise NPS breakdown
- `GET /api/nps/location` - Area-wise NPS analysis
- `GET /api/sentiment/location` - Location sentiment distribution
- `GET /api/analysis/summary` - Complete analysis overview

## 📈 Data Flow

1. **Startup**: Backend automatically loads `Data/reviews.json`
2. **AI Analysis**: Reviews processed through sentiment analysis & clustering
3. **Categorization**: Issues assigned to teams (UX/Dev/Payments) via LLM
4. **Caching**: Results stored for instant API responses
5. **Frontend**: Streamlit displays role-specific dashboards
6. **Real-time**: All endpoints serve live analyzed data

## 🎯 Use Cases

### For FSE Teams
- View department-specific customer issues
- Track team performance metrics
- Access clustered problem reports
- Monitor sentiment trends

### For Area Managers
- Regional performance overview
- Cross-team analytics
- Location-based insights
- Strategic decision support

### For Analysts
- Comprehensive review analysis
- NPS tracking across regions
- Sentiment analysis by location
- Automated issue categorization

## 🔧 Technical Stack

**Backend:**
- FastAPI (Web framework)
- PyJWT (Authentication)
- Sentence Transformers (Embeddings)
- Scikit-learn (Clustering)
- Google Gemini (LLM analysis)
- Transformers (Sentiment analysis)

**Frontend:**
- Streamlit (Web interface)
- Requests (API communication)

**AI/ML:**
- RoBERTa (Sentiment classification)
- K-Means (Issue clustering)
- Google Gemini (Issue categorization)
- Sentence Transformers (Text embeddings)

## 🚦 Getting Started

1. **Clone the repository**
2. **Set up backend** (install deps, configure API key, start server)
3. **Set up frontend** (install deps, start Streamlit)
4. **Access application** at `http://localhost:8501`
5. **Login** with demo credentials
6. **Explore** team-specific dashboards and analytics

## 📊 Sample Analysis Output

The system processes 40 customer reviews and provides:
- **Global NPS**: -62.5 (indicating areas for improvement)
- **Team Breakdown**: Issues categorized across UX, Dev, Payments
- **Location Analysis**: City-wise performance metrics
- **Clustered Issues**: Grouped similar complaints for action

## 🔒 Security Features

- JWT-based authentication
- Secure password handling
- CORS protection
- Environment variable configuration
- Session state management

## 🎨 UI Features

- Clean, modern interface
- Role-based dashboards
- Hidden sidebar for focus
- Interactive metrics display
- Real-time data updates

---

**Built with ❤️ using FastAPI, Streamlit, and AI-powered analytics**
