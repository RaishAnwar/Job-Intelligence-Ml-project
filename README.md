# Job Market Intelligence

Machine learning on ~85K Indian job postings: a **role classifier**, a **salary predictor** and a **resume-vs-job similarity scorer**, served through a **FastAPI** backend and a **Streamlit** app. All models are built with scikit-learn.

## Results at a glance

| Component | Result |
|---|---|
| Data | 97,929 raw postings cleaned to 85,613 |
| Role classifier (10 classes) | 63.9% accuracy, macro F1 0.57 (majority baseline: 22.8% accuracy) |
| Salary predictor | MAE 2.64 LPA vs 4.65 LPA for a median baseline (about 43% lower error) |
| Resume matcher | TF-IDF keyword similarity on skills (not evaluated against labels) |

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
- The cleaned file is saved as `data/processed/jobs_clean.csv` (not stored in this repository).

### EDA highlights

- Cities with the most postings: Bengaluru (14,949), Hyderabad (6,831), Pune (5,432), Chennai (3,872), Gurugram (3,810).
- Companies with the most postings: Accenture (5,620), IDESLABS PRIVATE LIMITED (1,905), Krazy Mantra HR Solutions (1,022), Wipro (866), Infosys (840).
- Median **minimum** salary among postings that disclose one: 3.00 LPA (mean 5.84 LPA).
- Most common titles: Application Developer (1,410), Application Lead (890), Sales Executive (495).
- The notebook also contains charts for the salary distribution, salary by years of experience, top locations and top companies.

## Role classifier (`notebooks/02_role_classifier.ipynb`)

**Task:** predict one of 10 role categories from the job description and skills.

**Labels:** the category is derived from the job title with keyword rules. The title