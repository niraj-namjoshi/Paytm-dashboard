Write-Host "Starting Streamlit with auto-reload..." -ForegroundColor Green
streamlit run app.py --server.port 8501 --server.fileWatcherType auto --server.runOnSave true
