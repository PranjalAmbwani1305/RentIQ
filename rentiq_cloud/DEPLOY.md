# Deploying RentIQ to Streamlit Cloud

## 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial RentIQ v3 commit"
git remote add origin https://github.com/<your-username>/rentiq.git
git push -u origin main
```

> ⚠️ `.streamlit/secrets.toml` is in `.gitignore` — never push it.

## 2. Create App on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Connect your GitHub repo
4. Set **Main file path** → `app.py`
5. Click **Advanced settings** → paste your secrets (see step 3)

## 3. Add Secrets
In Streamlit Cloud → **App settings → Secrets**, paste the contents of
`.streamlit/secrets.toml` exactly as-is.

## 4. Notes
- `artifacts/model.pkl` must be committed (it's ~2 MB, within GitHub's limit).
- The Spark page works in demo/simulation mode on Cloud (no JVM).
- The PyTorch deep-learning tab works but first-boot training takes ~2 min.
- Pinecone vector search requires a valid `PINECONE_API_KEY` in secrets.
