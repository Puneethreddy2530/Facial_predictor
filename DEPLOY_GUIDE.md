# 🚀 Deploy to Render.com (Free Tier)

This guide will help you deploy the AI Facial Insights backend to Render.com so it works from any device, not just localhost.

## 📋 Prerequisites

- GitHub account with your repository pushed
- Render.com account (free) - Sign up at https://render.com

## 🔧 Deployment Steps

### 1. **Sign up for Render.com**
   - Go to https://render.com
   - Click "Get Started for Free"
   - Sign up with your GitHub account (recommended)

### 2. **Create a New Web Service**
   - From your Render dashboard, click **"New +"** → **"Web Service"**
   - Connect your GitHub account if not already connected
   - Select your repository: `Puneethreddy2530/Facial_predictor`
   - Click **"Connect"**

### 3. **Configure the Service**
   Fill in these settings:

   | Setting | Value |
   |---------|-------|
   | **Name** | `facial-predictor-api` (or any name you prefer) |
   | **Region** | Choose closest to you |
   | **Branch** | `main` |
   | **Root Directory** | Leave empty |
   | **Runtime** | `Python 3` |
   | **Build Command** | `pip install --upgrade pip && pip install -r requirements.txt` |
   | **Start Command** | `uvicorn backend.main_deepface:app --host 0.0.0.0 --port $PORT` |
   | **Plan** | **Free** |

### 4. **Environment Variables** (Optional but recommended)
   Click **"Advanced"** and add these:
   
   - `PYTHON_VERSION` = `3.11.0`
   - `TF_CPP_MIN_LOG_LEVEL` = `2` (reduces TensorFlow logs)
   - `PYTHONUNBUFFERED` = `1`

### 5. **Deploy**
   - Click **"Create Web Service"**
   - Render will start building your app (this takes 5-10 minutes on first deploy)
   - Watch the logs to see progress
   - ⚠️ **First deployment is slow** because it downloads DeepFace models (~300MB)

### 6. **Get Your API URL**
   Once deployed, you'll see a URL like:
   ```
   https://facial-predictor-api-xxxx.onrender.com
   ```
   **Copy this URL** - you'll need it for the frontend!

## 🎯 Update Frontend to Use Your API

### Option 1: Use the GitHub Pages UI
1. Visit your GitHub Pages site: https://puneethreddy2530.github.io/Facial_predictor/
2. Paste your Render URL in the **"Backend API URL"** field (e.g., `https://facial-predictor-api-xxxx.onrender.com`)
3. Click **"Save"**
4. Upload a photo and click **"Analyze"** - it now works from any device! 🎉

### Option 2: Set Default API URL (recommended)
Update `frontend/script.js` to pre-fill your Render URL:

```javascript
const defaultApi = 'https://facial-predictor-api-xxxx.onrender.com';  // Replace with your URL
```

Then commit and push to auto-deploy to GitHub Pages.

## ⚠️ Important Notes

### Free Tier Limitations
- **Cold starts**: Free tier sleeps after 15 min of inactivity. First request after sleep takes ~30-60 seconds to wake up
- **750 hours/month**: Free tier gives you 750 hours (plenty for demos)
- **Shared CPU**: Performance is limited but sufficient for demos

### Optimize Performance
1. **Keep it warm**: Visit your API occasionally to prevent cold starts
2. **Use cron-job.org**: Set up a free ping every 10 minutes to keep it active
3. **Upgrade to paid**: $7/month removes cold starts and increases performance

## 🔍 Verify Deployment

Test your API is working:

```bash
# Check health endpoint
curl https://your-app.onrender.com/

# Test prediction (replace with your URL)
curl -X POST "https://your-app.onrender.com/predict" \
  -F "file=@path/to/your/image.jpg"
```

Or visit `https://your-app.onrender.com/docs` to see the interactive API documentation!

## 🐛 Troubleshooting

### Build Fails
- Check Render logs for errors
- Verify `requirements.txt` has all dependencies
- Ensure Python version is compatible (3.9-3.11 recommended)

### App Crashes
- Check runtime logs in Render dashboard
- DeepFace models might be downloading (wait 5-10 min)
- Free tier has 512MB RAM - might need optimization for very large images

### Slow First Request
- This is normal! Free tier "spins down" after inactivity
- First request wakes it up (~30-60 seconds)
- Subsequent requests are fast

### CORS Errors
- Already configured in `main_deepface.py` with `allow_origins=["*"]`
- Should work from any domain including GitHub Pages

## 🎉 Success!

Once deployed, your app works from **any device** with internet:
- ✅ Your laptop
- ✅ Friends' computers  
- ✅ Mobile phones
- ✅ Tablets
- ✅ Anywhere in the world!

No need to run `localhost:8001` anymore - just visit your GitHub Pages URL and it connects to your Render backend automatically!

---

## 🔄 Alternative: Railway.app

If Render doesn't work, try **Railway.app**:

1. Go to https://railway.app
2. Click "Start a New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects settings
5. Add environment variable: `PORT=8001`
6. Deploy!

Railway also has a free tier ($5/month credit).

---

## 📞 Need Help?

- Check Render logs: Dashboard → Your Service → Logs tab
- Render docs: https://render.com/docs
- GitHub Issues: Report problems in your repo

