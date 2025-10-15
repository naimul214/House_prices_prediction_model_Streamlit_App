# 🚀 Quick Start Guide

## Get Up and Running in 5 Minutes

### Step 1: Install Dependencies (1 minute)

```powershell
# Navigate to project directory
cd "d:\Coding Excercise\School\Assignment 2 - Deploying AI Systems"

# Create virtual environment (if not already done)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### Step 2: Verify Models (30 seconds)

Models are already pre-trained! Check:
```powershell
ls models/

# Should see:
# - model_lr.joblib
# - model_rf.joblib
# - area_encoder.joblib
# - city_encoder.joblib
# - scaler.joblib
```

If models are missing, train them:
```powershell
python -m src.train
# Wait 2-3 minutes
```

### Step 3: Run Tests (30 seconds)

```powershell
pytest tests/ -v
# Should see: 11 passed in ~1.5s
```

### Step 4: Launch App (30 seconds)

```powershell
streamlit run app.py
```

**App opens at**: http://localhost:8501

### Step 5: Test Features (2 minutes)

1. **Try Sample Data**:
   - Click "Predict" tab
   - Select "Use Sample Data"
   - Click "🎯 Predict Sample Prices"
   - View results

2. **Adjust Model**:
   - Sidebar → Select "Random Forest"
   - Adjust sliders
   - See changes

3. **View Visualizations**:
   - Click "Visualizations" tab
   - Try different chart types

4. **Download Results**:
   - After predictions, click "📥 Download Predictions"

---

## Common Commands

### Run Application
```powershell
streamlit run app.py
```

### Run Tests
```powershell
pytest tests/ -v
```

### Train Models
```powershell
python -m src.train
```

### Check Data
```powershell
python -c "import pandas as pd; df = pd.read_csv('properties.csv'); print(df.info())"
```

---

## Troubleshooting

### "Module not found" error
```powershell
pip install -r requirements.txt
```

### "Models not found" error
```powershell
python -m src.train
```

### Port already in use
```powershell
# Stop other Streamlit apps or use different port:
streamlit run app.py --server.port 8502
```

---

## What to Try

### 1. Upload Your Own Data
Create a CSV with columns: `Address`, `AreaName`, `lat`, `lng`

Example:
```csv
Address,AreaName,lat,lng
"123 Main St Toronto, ON",Downtown,43.65,-79.38
"456 Oak Ave Hamilton, ON",Central,43.25,-79.86
```

### 2. Experiment with Hyperparameters
- Sidebar → Random Forest Settings
- Try: n_estimators=200, max_depth=30
- Click "Retrain Models"
- Compare performance

### 3. Explore Visualizations
- Go to "Visualizations" tab
- Try each visualization type
- Hover over map points
- Check feature importance

### 4. Manual Prediction
- Predict tab → "Manual Input"
- Enter property details
- Select area from dropdown
- Get instant prediction with confidence interval

---

## File Locations

| File | Purpose |
|------|---------|
| `app.py` | Main application |
| `properties.csv` | Full dataset |
| `models/*.joblib` | Trained models |
| `src/` | Source code modules |
| `tests/` | Unit tests |
| `README.md` | Full documentation |

---

## Need Help?

- **Full Documentation**: See `README.md`
- **Component Details**: See `COMPONENTS_REFERENCE.md`
- **Deployment**: See `DEPLOYMENT_GUIDE.md`
- **Project Summary**: See `PROJECT_SUMMARY.md`

---

## Ready to Deploy?

See `DEPLOYMENT_GUIDE.md` for step-by-step instructions to deploy to Streamlit Cloud!

---

**Enjoy your Ontario House Price Predictor! 🏠📊**
