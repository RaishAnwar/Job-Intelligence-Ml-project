from fastapi import FastAPI
import os
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
app = FastAPI(
    title= "Job Marlet Intelligence API",
    version ="1.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..','models'))

salary_pipeline = joblib.load(os.path.join(MODELS_DIR,'salary_pipeline.pkl'))
print("Salary model loaded")
tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
role_classifier = joblib.load(os.path.join(MODELS_DIR, 'role_classifier_logreg.pkl'))
resume_matcher = joblib.load(os.path.join(MODELS_DIR, 'resume_matcher_pipeline.pkl'))
print("All models loaded successfully!")

# pydantic 
class SalaryInput(BaseModel):
    avg_experience: float
    location: str
    title:str= ""
    jobDescription: str = ""
    tagsAndSkills: str = ""

class RoleInput(BaseModel):
    title: str = ""
    jobDescription: str = ""
    tagsAndSkills: str = ""

class ResumeInput(BaseModel):
    resume_text: str
    job_description: str

@app.get("/")
def home():
    return {"message": "Job market API is running successfully!"}


@app.post("/predict_salary")
def predict_salary(data: SalaryInput):
    # User ke text fields ko combine kar rahe hain
    input_text = f"{data.title} {data.jobDescription} {data.tagsAndSkills}"
    
    # Model ke format ke hisab se DataFrame bana rahe hain
    input_df = pd.DataFrame({
        'avg_experience': [data.avg_experience],
        'location': [data.location],
        'text_features': [input_text]
    })
    
    try:
        # Model se prediction le rahe hain aur log scale ko LPA mein badal rahe hain
        log_pred = salary_pipeline.predict(input_df)[0]
        predicted_lpa = float(np.expm1(log_pred))
        
        return {
            "predicted_salary_lpa": round(predicted_lpa, 2)
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict_role")
def predict_role(data: RoleInput):
    try:
        # 1. Combine all text features into a single string (same as training time)
        text_features = f"{data.title} {data.jobDescription} {data.tagsAndSkills}"
        
        # 2. Transform the text into numbers using TF-IDF vectorizer
        text_vectorized = tfidf_vectorizer.transform([text_features])
        
        # 3. Predict the job role using the Logistic Regression model
        predicted_role = role_classifier.predict(text_vectorized)[0]
        
        return {
            "predicted_role": predicted_role
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/match_resume")
def match_resume(data: ResumeInput):
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Check if resume_matcher is directly a vectorizer or a dictionary
        if hasattr(resume_matcher, 'transform'):
            vectorizer = resume_matcher
        elif isinstance(resume_matcher, dict):
            vectorizer = None
            for key, val in resume_matcher.items():
                if hasattr(val, 'transform'):
                    vectorizer = val
                    break
            if vectorizer is None:
                return {"error": f"Could not find a vectorizer in dict keys: {list(resume_matcher.keys())}"}
        else:
            return {"error": f"Unsupported resume_matcher type: {type(resume_matcher)}"}

        # Transform both texts and calculate cosine similarity
        tfidf_matrix = vectorizer.transform([data.resume_text, data.job_description])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Return direct similarity score (0.0 to 1.0) rounded to 4 decimal places
        similarity = round(float(similarity_score), 4)

        return {
            "similarity": similarity
        }
    except Exception as e:
        return {"error": str(e)}