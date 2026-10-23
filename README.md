# Stroke Risk Prediction - MLOps Project (Healthcare)

An end-to-end **MLOps** project in the **healthcare** domain. Predicts whether a patient is likely to suffer a stroke from demographic, lifestyle and clinical features. Built for beginners learning MLOps: data validation, training, experiment comparison, model registry, serving (FastAPI), a Streamlit demo, monitoring (drift), tests and CI/CD.

## 1. Recommended Dataset (Kaggle)

**Primary recommendation:** [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) by fedesoriano

- **Link:** https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- **Size:** 5,110 patient records, 12 columns
- **Target:** `stroke` (binary: 1 = stroke, 0 = no stroke) - highly imbalanced (~4.9% positive)
- **Features:** gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status
- **License:** check the Kaggle page (open for research/education)
- **Why it's great for MLOps beginners:**
  - Healthcare domain with real-world impact (stroke = 2nd leading cause of death, WHO)
  - Mixed data types (numeric + categorical) -> feature pipelines
  - Missing values (BMI ~4%) -> imputation practice
  - Severe class imbalance -> threshold tuning, PR-AUC, monitoring
  - Small enough to iterate in seconds, large enough to be meaningful
  - Huge community: hundreds of public notebooks/tutorials

**Alternative (larger, multi-class):** [Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset) by prasad22 - 55,500 synthetic records, target `Test Results` (Normal/Abnormal/Inconclusive). Use this if you want a multi-class problem with richer feature engineering (dates, billing, hospitals).

## 2. Project Structure

```
stroke-risk-mlops/
├── data/                 # download + sample generation
├── src/stroke_risk/      # core package (data, features, models, predict, monitor)
├── api/                  # FastAPI service
├── app/                  # Streamlit demo
web/                  # Next.js premium frontend
├── scripts/              # train.py, evaluate.py
├── tests/                # pytest
├── notebooks/            # EDA
├── .github/workflows/    # CI/CD
├── Dockerfile, docker-compose.yml
└── requirements.txt
```

## 3. Quickstart (local)

```bash
# 1. Clone / enter project
cd stroke-risk-mlops

# 2. Create virtual env (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install deps
pip install -r requirements.txt

# 4. Get data
# Option A - synthetic sample (no Kaggle creds, good for CI/demo):
python data/generate_sample.py

# Option B - real Kaggle data:
#   pip install kaggle  (and put kaggle.json in ~/.kaggle/)
python data/download_data.py
# or download manually from the Kaggle link above and place at:
#   data/raw/healthcare-dataset-stroke-data.csv

# 5. Train
python scripts/train.py

# 6. Evaluate
python scripts/evaluate.py

# 7. Run tests
pytest -q

# 8. Serve API
uvicorn api.main:app --reload --port 8000
# docs: http://127.0.0.1:8000/docs

# 9. Streamlit demo
streamlit run app/streamlit_app.py

# 10. Next.js premium frontend (web/)
cd web
npm install
npm run dev
# open http://localhost:3000 (or 3001)
```

## 4. API Usage

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender":"Male","age":67,"hypertension":0,"heart_disease":1,
    "ever_married":"Yes","work_type":"Private","Residence_type":"Urban",
    "avg_glucose_level":228.69,"bmi":36.6,"smoking_status":"formerly smoked"
  }'
```

## 5. MLOps Concepts Covered

| MLOps area | Implementation |
|---|---|
| Data validation | `src/stroke_risk/data.py::validate` (schema, missing, duplicates, target rate) |
| Reproducible splits | stratified train/test, fixed seed, saved to `data/processed/` |
| Feature pipeline | scikit-learn `ColumnTransformer` + `Pipeline` (impute + scale + one-hot) |
| Model selection | 3 candidates compared via stratified CV (avg precision) |
| Threshold tuning | recall-constrained threshold for medical screening |
| Model registry | versioned artifacts in `models/` + `model_metadata.json` |
| Serving | FastAPI with `/predict`, `/predict/batch`, `/health`, `/metadata` |
| Demo UI | Streamlit app |
| Monitoring | PSI-based drift detection per feature (`monitor.py`) |
| Testing | pytest (data, features, API, monitoring) |
| CI/CD | GitHub Actions (test on push, docker on release) |
| Containerization | Dockerfile + docker-compose |

## 6. Publishing Your Project

### A. GitHub
```bash
git init
git add .
git commit -m "feat: stroke risk mlops project"
git branch -M main
git remote add origin https://github.com/<you>/stroke-risk-mlops.git
git push -u origin main
```
The included CI workflow runs tests on every push.

### B. Docker (local registry / Docker Hub)
```bash
docker build -t <you>/stroke-risk-mlops:latest .
docker run -p 8000:8000 <you>/stroke-risk-mlops:latest
# push:
docker push <you>/stroke-risk-mlops:latest
```

### C. Streamlit Cloud (free demo)
1. Push to GitHub.
2. Go to https://share.streamlit.io -> New app -> select repo -> `app/streamlit_app.py`.
3. Add `requirements.txt` (already present).
4. Note: model artifact must be available; commit `models/*.joblib` (remove from .gitignore) OR train on startup.

### D. Hugging Face Spaces
- Create a Space (Docker or Streamlit SDK).
- Upload repo; Spaces auto-builds from Dockerfile.

### E. Cloud (AWS/GCP/Azure)
- Build container, push to ECR/GCR/ACR, deploy to ECS/Cloud Run/App Service.
- Add a model registry (MLflow) and a feature store for production scale.

## 7. Extending (next steps for learning)

- Add **MLflow** for experiment tracking: `pip install mlflow`, log params/metrics/artifacts in `train.py`.
- Add **DVC** for data versioning: `dvc init`, `dvc add data/raw/...`.
- Add **Great Expectations** / **Evidently** for richer data quality & drift reports.
- Add **GitHub Actions CD** to auto-deploy container on release (fill secrets).
- Add **model monitoring** dashboard (drift over time, prediction distribution).
- Try the **Healthcare Dataset** (prasad22) for a multi-class variant.
- Add **explainability** with SHAP.

## 8. Notes on Data & Ethics

- The synthetic sample is for development/CI only; use the real Kaggle data for real experiments.
- Healthcare ML must not be used for actual clinical decisions without validation, regulatory review and bias audits.
- Respect the Kaggle dataset license and terms.

## 9. License

MIT (code). Dataset license per Kaggle page.