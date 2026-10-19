# Job Market Intelligence

Machine learning on ~85K Indian job postings: a **role classifier**, a **salary predictor** and a **resume-vs-job similarity scorer**, served through a **FastAPI** backend and a **Streamlit** app. All models are built with scikit-learn.

## Results at a glance

| Component | Result |
|---|---|
| Data | 97,929 raw postings cleaned to 85,613 |
| Role classifier (10 classes) | 63.9% accuracy, macro F1 0.57 (majority baseline: 22.8% accuracy) |
| Salary predictor | MAE 2.64 LPA vs 4.65 LPA for a median baseline (about 43% lower error) |
| Resume matcher | TF-IDF keyword similarity (not evaluated against labels) |

## What it does

| Feature | Input | Output |
|---|---|---|
| Role classifier | Job description + skills | One of 10 role categories |
| Salary predictor | Experience, location, title, description, skills | Expected salary in LPA |
| Resume matcher | Resume text + job description | TF-IDF similarity score (0 to 1) |

## Dataset

- File: `indian-job-market-dataset-2025.xlsx` (Indian job postings, 97,929 rows, 17 columns).
- Columns include title, company, location, experience, salary range, skills, job description, company rating and review count.
- The raw data is not stored in this repository. To re-run the notebooks, download the dataset and place it in `data/raw/`.

### Data cleaning (`notebooks/01_eda.ipynb`)

| Step | Rows left |
|---|---|
| Raw dataset | 97,929 |
| Keep INR postings only (129 USD rows removed) | 97,800 |
| Drop rows with missing experience or skills | 97,229 |
| Remove duplicate `jobId` and reposted jobs (same title, company and description) | **85,613** |

- "Not disclosed" salaries are stored as `0.0` in the raw data. They were converted to missing values, leaving 30,024 of 85,613 postings (35%) with a disclosed salary.
- HTML tags were removed from job descriptions.

### EDA highlights

- Cities with the most postings: Bengaluru (14,949), Hyderabad (6,831), Pune (5,432), Chennai (3,872), Gurugram (3,810).
- Companies with the most postings: Accenture (5,620), IDESLABS PRIVATE LIMITED (1,905), Krazy Mantra HR Solutions (1,022), Wipro (866), Infosys (840).
- Median **minimum** salary among postings that disclose one: 3.00 LPA (mean 5.84 LPA).
- Most common titles: Application Developer (1,410), Application Lead (890), Sales Executive (495).

## Role classifier (`notebooks/02_role_classifier.ipynb`)

**Task:** predict one of 10 role categories from the job description and skills.

**Labels:** the category is derived from the job title with keyword rules. The title itself is **not** a model input, to avoid target leakage. Input text = `jobDescription + tagsAndSkills`.

**Setup:** 68,490 training rows and 17,123 test rows (80/20, stratified, `random_state=42`). TF-IDF (10,000 features, 1-2 grams, English stop words) is fitted on the training set only.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Majority-class baseline (Software & Tech) | 22.84% | ~0.04 |
| **Logistic Regression (final)** | **63.90%** | **0.57** |
| LinearSVC (balanced class weights) | 62.20% | 0.57 |

Logistic Regression was chosen because its accuracy and macro precision are higher, macro F1 is tied, it is simple, and it provides `predict_proba`.

### Per-class results (Logistic Regression, test set)

| Role category | Precision | Recall | F1 | Test jobs |
|---|---|---|---|---|
| Software & Tech | 0.73 | 0.86 | 0.79 | 3,911 |
| Product & Design | 0.74 | 0.70 | 0.72 | 175 |
| Sales & BD | 0.65 | 0.75 | 0.70 | 3,760 |
| Data & AI | 0.63 | 0.56 | 0.59 | 2,502 |
| HR & Talent | 0.67 | 0.51 | 0.58 | 656 |
| Marketing & Media | 0.56 | 0.55 | 0.56 | 2,093 |
| Finance & Banking | 0.60 | 0.50 | 0.55 | 1,522 |
| Others / General | 0.50 | 0.42 | 0.46 | 1,818 |
| Operations & Admin | 0.53 | 0.36 | 0.43 | 368 |
| Customer Support | 0.42 | 0.31 | 0.36 | 318 |

### Error analysis (confusion matrix)

- Of 2,502 Data & AI test jobs, 1,396 were classified correctly. The main confusions: 354 predicted as Software & Tech, 285 as Sales & BD and 161 as Marketing & Media.
- Software & Tech and Sales & BD are the largest classes and absorb many mistakes from smaller classes.
- Customer Support, Operations & Admin and Others / General are the weakest classes.
- The Data & AI jobs predicted as Sales & BD were not inspected in detail, so the cause is not confirmed.

## Salary predictor (`notebooks/03_salary_predictor.ipynb`)

**Task:** predict the average of the minimum and maximum salary, in LPA.

**Data:** 29,602 postings with a disclosed salary between 1 and 100 LPA (23,681 train, 5,921 test, random 80/20 split, `random_state=42`).

**Features:**
- `avg_experience` (mean of minimum and maximum experience)
- `location` (one-hot encoded, up to 30 categories)
- `text_features` = title + job description + skills (TF-IDF, 500 features)

**Target:** `log1p(salary)`. Predictions are converted back to LPA with `expm1`.

| Model | MAE (LPA) | RMSE (LPA) | R² (LPA scale) |
|---|---|---|---|
| Median baseline | 4.65 | n/a | n/a |
| Ridge Regression | 2.85 | 6.45 | 0.480 |
| **HistGradientBoosting (final)** | **2.64** | **6.08** | **0.540** |

- The final model's MAE is about 43% lower than the median baseline (4.65 to 2.64 LPA).
- Adding the job title to the text features improved MAE slightly (2.70 to 2.64 LPA) and R² on the log scale from 0.692 to 0.705.

### Error by salary range (test set)

| Actual salary | Jobs | Avg actual (LPA) | Avg predicted (LPA) | MAE (LPA) |
|---|---|---|---|---|
| 0-5 | 3,602 | 3.16 | 3.76 | 0.99 |
| 5-10 | 1,094 | 6.83 | 7.01 | 2.12 |
| 10-20 | 774 | 14.08 | 12.85 | 4.38 |
| 20-40 | 368 | 27.25 | 18.85 | 9.44 |
| 40+ | 83 | 55.83 | 21.13 | 34.70 |

The model is reasonably stable up to 20 LPA, where about 92% of test jobs sit. It under-predicts high salaries badly. The 40+ bucket has only 83 jobs, so that row is noisy.

## Resume matcher

- A TF-IDF vectorizer (unigrams, English stop words, 14,550-term vocabulary) is fitted on the skills of all 85,613 postings.
- The resume and the job description are vectorized and compared with cosine similarity (0 = nothing in common, 1 = identical).
- The score measures keyword overlap, not meaning. It is **not** a match percentage, so use it to compare resumes against each other.
- The matcher has not been evaluated against ground-truth labels.

## Saved models (`models/`)

| File | Contents |
|---|---|
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer for the role classifier |
| `role_classifier_logreg.pkl` | Logistic Regression role classifier |
| `salary_pipeline.pkl` | Salary pipeline (preprocessing + HistGradientBoosting) |
| `resume_matcher_pipeline.pkl` | Matcher TF-IDF vectorizer, skills matrix and job table |

## API (FastAPI)

| Method | Endpoint | Body | Response |
|---|---|---|---|
| GET | `/` | none | status message |
| POST | `/predict_salary` | `avg_experience`, `location`, `title`, `jobDescription`, `tagsAndSkills` | `{"predicted_salary_lpa": ...}` |
| POST | `/predict_role` | `jobDescription`, `tagsAndSkills` | `{"predicted_role": "..."}` |
| POST | `/match_resume` | `resume_text`, `job_description` | `{"similarity": ...}` |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict_salary \
  -H "Content-Type: application/json" \
  -d '{"avg_experience": 4, "location": "Hyderabad", "title": "Data Scientist", "jobDescription": "Build ML models with Python.", "tagsAndSkills": "Python, Machine Learning, SQL"}'
```

Interactive docs are available at `http://127.0.0.1:8000/docs` once the server is running.

## Streamlit app

Three pages in the sidebar: **Salary Intelligence**, **Role Classifier** and **Resume Matcher**. The app calls the FastAPI backend, so start the API first.

## Project structure

```
job_intel/
  api/
    main.py                      FastAPI backend
  app/
    app.py                       Streamlit app
  data/
    raw/                         original dataset (not in repo)
    processed/                   jobs_clean.csv (not in repo)
  models/
    role_classifier_logreg.pkl
    tfidf_vectorizer.pkl
    salary_pipeline.pkl
    resume_matcher_pipeline.pkl
  notebooks/
    01_eda.ipynb                 cleaning and EDA
    02_role_classifier.ipynb
    03_salary_predictor.ipynb
  requirements.txt
  README.md
```

## Run locally

```bash
git clone https://github.com/RaishAnwar/Job-Intelligence-Ml-project
cd Job-Intelligence-Ml-project
python -m venv venv
venv\Scripts\activate          # Windows (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

# terminal 1: API
uvicorn api.main:app --reload

# terminal 2: app
streamlit run app/app.py
```

Developed on Windows with Python 3.14 and scikit-learn 1.9.x. Use the versions in `requirements.txt` when loading the saved models.

## Limitations

- **Role labels are noisy.** They come from keyword rules on job titles, and some rules use plain substring matching. Short keywords such as "ai" match titles like "Maintenance Engineer" or "Trainer". About 36% of the Data & AI labels came only from substring matches of "ai", "sql" or "etl" (a few, such as "PLSQL Developer", are legitimate). This limits classifier accuracy.
- **Salary is disclosed for only 35% of postings**, and the model is trained on that subset only.
- **The salary model under-predicts salaries above 20 LPA** and is more reliable for junior and mid-level roles. One random train/test split was used, with no cross-validation.
- **Salary text check:** about 6% of training texts contain words like "lpa" or "lakh". Whether this leaks salary information into the model was not tested.
- **Location is a raw string** (for example "Hyderabad, Bengaluru" is one category), and the dataset spells "Bengaluru", not "Bangalore".
- **Resume matching is keyword-based** (TF-IDF), not semantic.
- The data is a single snapshot from 2025.

## Future work

- Fix the title keyword rules (word-boundary matching) and retrain the role classifier.
- Split multi-city locations and add company features such as rating and review count.
- Predict a salary range with quantile regression.
- Use sentence embeddings for resume matching and add a top-N job recommendation endpoint.
- Add cross-validation and an evaluation of the matcher.

## Tech stack

Python, pandas, NumPy, scikit-learn, joblib, FastAPI, Uvicorn, Streamlit, Matplotlib, Seaborn.

## Author

Raish