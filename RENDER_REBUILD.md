# 🔄 How to Force Render Rebuild

If you see the tf-keras error in Render logs, here's how to force a rebuild:

## Method 1: Manual Deploy (Fastest - 2 clicks)
1. Go to https://dashboard.render.com
2. Find your **facial-predictor-api** service
3. Click **"Manual Deploy"** → **"Deploy latest commit"**
4. Wait 5-10 minutes for rebuild to complete

## Method 2: Empty Commit (Automatic)
```powershell
git commit --allow-empty -m "Trigger rebuild"
git push origin main
```
Render will auto-detect and rebuild.

## Method 3: Clear Build Cache (Nuclear Option)
1. Go to https://dashboard.render.com
2. Find your service
3. Click **Settings**
4. Scroll to **"Danger Zone"**
5. Click **"Clear build cache"**
6. Then click **"Manual Deploy"**

## 🔍 How to Check Rebuild Status

1. Go to https://dashboard.render.com
2. Click on **facial-predictor-api**
3. Click **"Events"** tab
4. Look for:
   - "Deploy started"
   - "Build started"
   - "Installing dependencies" (should show tf-keras installing)
   - "Deploy live"

## ✅ How to Verify It's Fixed

After rebuild completes:

**Test 1: Health Check**
```powershell
Invoke-WebRequest -Uri "https://facial-predictor-api.onrender.com/" -UseBasicParsing
```
Should return: `{"status":"healthy"...}`

**Test 2: Prediction Test**
Visit: https://puneethreddy2530.github.io/Facial_predictor/test-api.html

Upload an image and check the response. If no "tf-keras" error, it's fixed!

## 📊 Current Status

Run this to check if tf-keras is in requirements.txt:
```powershell
Get-Content requirements.txt | Select-String -Pattern "keras"
```

Should show: `tf-keras` ✅

## ⏱️ Expected Timeline

- **Commit pushed**: ✅ Done (just now)
- **Render detects change**: 1-2 minutes
- **Build starts**: Immediately after detection
- **Dependencies install**: 5-7 minutes
- **Deploy completes**: 8-10 minutes total

## 🐛 If Still Not Working

1. Check Render logs for new errors
2. Verify requirements.txt was pushed to GitHub
3. Try Method 1 (Manual Deploy)
4. Check if Render free tier has limitations

---

**Pro Tip**: Render auto-deploys when you push to GitHub. The rebuild we just triggered should fix the tf-keras issue automatically!
