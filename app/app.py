import streamlit as st
import requests

# FastAPI Backend URL
BACKEND_URL = "http://127.0.0.1:8000"

# Page Configuration
st.set_page_config(
    page_title="Job Intel & Career AI Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING FOR A PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    /* Main Background & Font Styling */
    .main {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Card Containers */
    .metric-card {
        background-color: #21262d;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* Headers and Subheaders */
    h1, h2, h3 {
        color: #f0f6fc !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Custom Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #238636 100%);
        box-shadow: 0 0 10px rgba(46, 160, 67, 0.5);
    }

    /* Input Fields Border Fix */
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #0d1117;
        color: #f0f6fc;
        border: 1px solid #30363d;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION & APP HEADER ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=100)
    st.title("Job Intel Hub")
    st.markdown("---")
    
    # Styled radio for navigation
    tab = st.radio(
        "Navigation Menu", 
        ["💰 Salary Intelligence", "🏷️ Role Classifier", "📄 Resume Matcher"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### System Status")
    try:
        health_check = requests.get(f"{BACKEND_URL}/", timeout=2)
        if health_check.status_code == 200:
            st.success("🟢 Backend API: Online")
        else:
            st.warning("🟡 Backend API: Issues Detected")
    except:
        st.error("🔴 Backend API: Offline (Run Uvicorn)")

# --- TAB 1: SALARY PREDICTOR ---
if tab == "💰 Salary Intelligence":
    st.markdown("# 💰 Salary Prediction Intelligence")
    st.markdown("Estimate market-accurate compensation packages based on your experience, location, skills, and job description.")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            avg_experience = st.number_input("Average Experience (Years)", min_value=0.0, max_value=30.0, value=3.0, step=0.5, help="Total professional experience in years.")
            location = st.text_input("Job Location", value="Bangalore", help="Target city or remote hub.")
        with col2:
            title = st.text_input("Job Title / Role", value="Data Scientist", help="e.g., Senior Data Scientist, Software Engineer")
            tagsAndSkills = st.text_input("Core Skills (Comma separated)", value="Python, Machine Learning, SQL, PyTorch", help="Key technical skills required.")

        jobDescription = st.text_area("Full Job Description / Requirements", value="Looking for an experienced professional to manage complex data pipelines, fine-tune machine learning models, and drive data-driven decision making.", height=120)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Calculate Expected Salary"):
            with st.spinner("Analyzing market trends and predicting salary..."):
                payload = {
                    "avg_experience": avg_experience,
                    "location": location,
                    "title": title,
                    "jobDescription": jobDescription,
                    "tagsAndSkills": tagsAndSkills
                }
                try:
                    response = requests.post(f"{BACKEND_URL}/predict_salary", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        salary = result.get('predicted_salary_lpa')
                        
                        # Styled Result Card Container
                        st.markdown(f"""
                            <div style="background-color: #1f6feb22; border: 1px solid #1f6feb; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
                                <h3 style="color: #58a6ff !important; margin: 0;">Estimated Package</h3>
                                <h1 style="color: #ffffff !important; font-size: 42px; margin: 10px 0;">{salary} LPA</h1>
                                <p style="color: #8b949e; margin: 0;">Predicted based on current market features and input parameters.</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection failed: Ensure FastAPI backend is running. Details: {e}")

# --- TAB 2: ROLE CLASSIFIER ---
elif tab == "🏷️ Role Classifier":
    st.markdown("# 🏷️ Industry Domain & Role Classifier")
    st.markdown("Automatically categorize job postings into their respective industry sectors using NLP and classification models.")

    with st.container():
        title_role = st.text_input("Job Title", value="Senior Machine Learning Engineer")
        desc_role = st.text_area("Job Description Summary", value="Build and deploy generative AI applications, fine-tune large language models, and scale vector search architectures.", height=120)
        skills_role = st.text_input("Tags & Associated Skills", value="Python, PyTorch, LangChain, LLMs, Vector Databases")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Classify Job Category"):
            with st.spinner("Analyzing text content and classifying domain..."):
                payload = {
                    "title": title_role,
                    "jobDescription": desc_role,
                    "tagsAndSkills": skills_role
                }
                try:
                    response = requests.post(f"{BACKEND_URL}/predict_role", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        role_category = result.get('predicted_role')
                        
                        st.markdown(f"""
                            <div style="background-color: #23863622; border: 1px solid #238636; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
                                <h3 style="color: #3fb950 !important; margin: 0;">Classified Industry Domain</h3>
                                <h1 style="color: #ffffff !important; font-size: 36px; margin: 10px 0;">{role_category}</h1>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

# --- TAB 3: RESUME MATCHER ---
elif tab == "📄 Resume Matcher":
    st.markdown("# 📄 Semantic Resume Matcher")
    st.markdown("Compute the semantic similarity score between candidate resume text and target job descriptions using high-dimensional vector spaces.")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            resume_text = st.text_area("Candidate Resume Text", value="Experienced Data Scientist skilled in Python, Machine Learning, Pandas, NumPy, Scikit-learn, and FastAPI backend services.", height=200)
        with col2:
            job_description = st.text_area("Target Job Description", value="We are hiring a Data Scientist proficient in Python, Machine Learning, and SQL to build scalable pipelines.", height=200)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Calculate Semantic Similarity"):
            with st.spinner("Computing cosine similarity score..."):
                payload = {
                    "resume_text": resume_text,
                    "job_description": job_description
                }
                try:
                    response = requests.post(f"{BACKEND_URL}/match_resume", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        similarity = result.get('similarity')
                        
                        # Score interpretation logic
                        if similarity >= 0.6:
                            color, status = "#3fb950", "High Match 🚀"
                        elif similarity >= 0.3:
                            color, status = "#d29922", "Moderate Match ⚡"
                        else:
                            color, status = "#f85149", "Low Match ⚠️"

                        st.markdown(f"""
                            <div style="background-color: {color}22; border: 1px solid {color}; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
                                <h3 style="color: {color} !important; margin: 0;">Similarity Score: {status}</h3>
                                <h1 style="color: #ffffff !important; font-size: 42px; margin: 10px 0;">{similarity}</h1>
                                <p style="color: #8b949e; margin: 0;">Score range spans from 0.0 (no match) to 1.0 (exact semantic match).</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Progress bar representation
                        st.progress(float(similarity))
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")