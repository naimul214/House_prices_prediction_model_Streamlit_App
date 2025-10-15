# Deployment Guide - Ontario House Price Predictor

## 🚀 Quick Deployment to Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at share.streamlit.io)
- This repository pushed to GitHub

---

## Step-by-Step Deployment

### 1️⃣ Prepare Your Repository

#### A. Initialize Git (if not already done)
```powershell
cd "d:\Coding Excercise\School\Assignment 2 - Deploying AI Systems"
git init
git add .
git commit -m "Initial commit: Ontario House Price Predictor"
```

#### B. Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `ontario-house-price-predictor` (or your choice)
3. Description: "ML application for predicting Ontario house prices"
4. Keep it **Public** (required for free Streamlit Cloud)
5. Don't initialize with README (you already have one)
6. Click "Create repository"

#### C. Push to GitHub
```powershell
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ontario-house-price-predictor.git
git push -u origin main
```

⚠️ **Important**: Make sure these files are committed:
- ✅ `app.py`
- ✅ `requirements.txt`
- ✅ `properties.csv`
- ✅ All files in `src/` directory
- ✅ All files in `models/` directory (pre-trained models)
- ✅ `.streamlit/config.toml`

---

### 2️⃣ Deploy to Streamlit Cloud

#### A. Sign Up/Login
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Authorize Streamlit to access your repositories

#### B. Create New App
1. Click **"New app"** button (top right)
2. Fill in the form:

```
Repository: YOUR_USERNAME/ontario-house-price-predictor
Branch: main
Main file path: app.py
App URL (optional): ontario-house-prices (or leave default)
```

3. Click **"Advanced settings"** (optional)
   - Python version: 3.10 (recommended)
   - No secrets needed for this app

4. Click **"Deploy!"**

#### C. Wait for Build
- Initial build: 3-5 minutes
- Watch the build logs for any errors
- Look for: "Your app is now running!"

#### D. Test Your App
1. Click the app URL (e.g., `ontario-house-prices.streamlit.app`)
2. Test all features:
   - ✅ Model selection works
   - ✅ CSV upload works
   - ✅ Manual prediction works
   - ✅ Sample data works
   - ✅ Visualizations render
   - ✅ Download button works

---

### 3️⃣ Update README with Live URL

Once deployed, update your README:

```powershell
# Edit README.md line 8
# Change:
**Deployed Application**: [TBD - Add Streamlit Cloud URL here]

# To:
**Deployed Application**: [https://ontario-house-prices.streamlit.app](https://ontario-house-prices.streamlit.app)

# Commit and push
git add README.md
git commit -m "Add deployed app URL to README"
git push
```

---

## 🔧 Troubleshooting

### Problem: Build Fails with "ModuleNotFoundError"

**Solution**: Check `requirements.txt` has all dependencies
```
streamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
joblib>=1.3.0
pytest>=7.4.0
```

### Problem: "File not found: properties.csv"

**Solution**: Ensure `properties.csv` is committed to git
```powershell
git add properties.csv
git commit -m "Add dataset"
git push
```

### Problem: "Models not found" error on startup

**Solution 1**: Commit pre-trained models (recommended)
```powershell
git add models/
git commit -m "Add pre-trained models"
git push
```

**Solution 2**: Let app train on first run
- Click "Retrain Models" in sidebar
- Wait 2-3 minutes for training
- Models will be saved to Streamlit Cloud's temporary storage
- Note: Models will reset if app restarts

### Problem: App is slow or crashes

**Possible Causes & Solutions**:

1. **Large dataset**: 
   - App already samples data for visualizations (1000 points max)
   - Consider reducing `properties.csv` size if needed

2. **Memory limit exceeded**:
   - Streamlit Cloud has 1GB RAM limit
   - Remove unnecessary data columns
   - Use more efficient data types (int16 instead of int64)

3. **Too many requests**:
   - Add caching decorators (already implemented)
   - Limit concurrent users with rate limiting

### Problem: Visualizations don't appear

**Solution**: Check Plotly is installed
```
# In requirements.txt
plotly>=5.17.0
```

### Problem: App shows stale data after updates

**Solution**: Clear cache
1. In Streamlit Cloud dashboard, click "..." on your app
2. Select "Reboot app"
3. Or: Click "Clear cache" in the app's hamburger menu

---

## 📊 Monitoring Your Deployed App

### View Logs
1. Go to Streamlit Cloud dashboard
2. Click on your app
3. View "Logs" tab for errors and activity

### View Analytics
1. In the app, click hamburger menu (☰)
2. Select "Analytics"
3. View:
   - Page views
   - Unique visitors
   - Session duration
   - Error rates

### Update Your App
```powershell
# Make changes locally
# Test locally: streamlit run app.py

# Commit and push
git add .
git commit -m "Description of changes"
git push

# Streamlit Cloud auto-redeploys (2-3 minutes)
```

---

## 🔐 Security Best Practices

### Public Deployment (Current Setup)
✅ No sensitive data in code
✅ No API keys or credentials
✅ Dataset is public information
✅ No user authentication needed

### If Adding Secrets (Future)
1. In Streamlit Cloud dashboard → App settings → Secrets
2. Add key-value pairs
3. Access in code:
```python
import streamlit as st
api_key = st.secrets["api_key"]
```

---

## 🎨 Custom Domain (Optional)

Streamlit Cloud provides: `your-app.streamlit.app`

For custom domain (e.g., `predict.yourdomain.com`):
1. Upgrade to Streamlit Cloud paid plan
2. Follow custom domain setup guide
3. Update DNS records with your registrar

---

## 📈 Performance Optimization

### Already Implemented
✅ Caching with `@st.cache_data` and `@st.cache_resource`
✅ Pre-trained models (no training on load)
✅ Data sampling for visualizations
✅ Efficient data types

### Additional Optimizations
1. **Reduce CSV size**: Keep only essential columns
2. **Compress models**: Use model compression techniques
3. **Lazy loading**: Load data only when needed
4. **Pagination**: For large result sets
5. **CDN for static assets**: For images, fonts

---

## 🔄 CI/CD Integration (Advanced)

### GitHub Actions for Auto-Testing

Create `.github/workflows/test.yml`:

```yaml
name: Test and Deploy
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest tests/ -v
      - name: Check if tests pass
        if: success()
        run: echo "Tests passed! Streamlit Cloud will auto-deploy."
```

This runs tests on every push and pull request.

---

## 📱 Mobile Optimization

The app is responsive, but for better mobile UX:

1. **Test on mobile**: Use browser dev tools or real device
2. **Adjust layouts**: Consider mobile-first design
3. **Simplify inputs**: Use fewer columns on small screens
4. **Touch-friendly**: Ensure buttons/sliders are easily tappable

---

## 🌍 Making Your App Public

### Share Your App
- **Direct link**: Share the Streamlit Cloud URL
- **QR code**: Generate QR code for the URL
- **Social media**: Post screenshots and link
- **Portfolio**: Add to your GitHub profile README

### Example README Badge
```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
```

---

## 📞 Getting Help

### Resources
- **Streamlit Docs**: https://docs.streamlit.io
- **Community Forum**: https://discuss.streamlit.io
- **GitHub Issues**: For bug reports
- **Stack Overflow**: Tag with [streamlit]

### Common Questions

**Q: How much does Streamlit Cloud cost?**
A: Free tier includes public apps with reasonable usage limits. Paid tiers for private apps and higher limits.

**Q: Can I use a database?**
A: Yes! Streamlit supports connections to PostgreSQL, MySQL, MongoDB, and more.

**Q: How do I add authentication?**
A: Use streamlit-authenticator package or integrate OAuth.

**Q: Can I schedule tasks (e.g., retrain models)?**
A: Not directly. Use external schedulers (GitHub Actions, cron) or cloud functions.

---

## ✅ Deployment Checklist

Before going live:

- [ ] All tests passing locally
- [ ] README has clear description and instructions
- [ ] requirements.txt is complete and tested
- [ ] Dataset is included or downloadable
- [ ] Models are pre-trained and committed
- [ ] .gitignore excludes sensitive files
- [ ] App runs smoothly on local machine
- [ ] Error handling covers edge cases
- [ ] Help/About sections are informative
- [ ] App URL is memorable (if customized)
- [ ] README has live app URL
- [ ] Screenshots added to README (optional)
- [ ] App is tested on multiple devices
- [ ] Analytics are set up (if needed)

---

## 🎉 Post-Deployment

### Announce Your App
- Update your LinkedIn/Twitter
- Share in relevant communities
- Add to your portfolio website
- Submit to Streamlit Gallery (https://streamlit.io/gallery)

### Maintain Your App
- Monitor logs for errors
- Update dependencies periodically
- Add new features based on feedback
- Keep dataset current (if applicable)

---

**Congratulations! Your app is now live! 🚀**

For questions or issues, refer to the main README or open an issue on GitHub.

---

*Last Updated: October 15, 2025*
