import streamlit as st
import sqlite3
import pandas as pd
import json
import io
import re
import numpy as np

# NEW: Authentication Library
import bcrypt 

# NLP and Utility Libraries
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
from pdfminer.high_level import extract_text_to_fp

# --- Global Configurations & Initialization ---
DB_PATH = "resume_analyzer.db"
# Use environment variables for app ID and Firebase config if available
APP_ID = "resume_analyzer_app" # Fixed ID for demonstration purposes
# FIREBASE_CONFIG is not used in this purely local SQLite demo.

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Error message remains, critical for local setup
    st.error("Error loading spaCy model. Please ensure 'en_core_web_sm' is downloaded ('python -m spacy download en_core_web_sm').")
    nlp = None

# --- Authentication and Database Functions ---

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def setup_database():
    """Creates the necessary tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    TABLE_PREFIX = f"{APP_ID}_"

    # NEW: users table with secure hashed_password
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        role TEXT NOT NULL, -- 'student' or 'recruiter'
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Existing tables, modified to include user_id/recruiter_id for filtering
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}resumes (
        resume_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, -- FK to users
        text TEXT,
        parsed_json TEXT,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}job_descriptions (
        jd_id INTEGER PRIMARY KEY AUTOINCREMENT,
        recruiter_id INTEGER, -- FK to users
        text TEXT,
        parsed_json TEXT,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}results (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        resume_id INTEGER,
        jd_id INTEGER,
        ats_score REAL,
        match_details TEXT,
        suggestions TEXT,
        FOREIGN KEY(resume_id) REFERENCES {TABLE_PREFIX}resumes(resume_id),
        FOREIGN KEY(jd_id) REFERENCES {TABLE_PREFIX}job_descriptions(jd_id)
    )
    """)
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hashes a plaintext password using bcrypt."""
    # Salt is automatically generated and embedded in the hash
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed_password):
    """Checks a plaintext password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    except ValueError:
        return False # Handle incorrect hash format

def register_user(name, email, password, role):
    """Registers a new user into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    TABLE_PREFIX = f"{APP_ID}_"
    
    try:
        hashed_pwd = hash_password(password)
        cursor.execute(f"""
        INSERT INTO {TABLE_PREFIX}users (name, email, hashed_password, role) 
        VALUES (?, ?, ?, ?)
        """, (name, email, hashed_pwd, role))
        conn.commit()
        return True, "Registration successful! Please log in."
    except sqlite3.IntegrityError:
        return False, "This email is already registered."
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"
    finally:
        conn.close()

def authenticate_user(email, password):
    """Authenticates user credentials against the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    TABLE_PREFIX = f"{APP_ID}_"
    
    cursor.execute(f"SELECT user_id, name, hashed_password, role FROM {TABLE_PREFIX}users WHERE email = ?", (email,))
    user_record = cursor.fetchone()
    conn.close()
    
    if user_record:
        if check_password(password, user_record['hashed_password']):
            return {
                'logged_in': True,
                'user_id': user_record['user_id'],
                'user_name': user_record['name'],
                'user_role': user_record['role']
            }
    return {'logged_in': False}

def logout():
    """Clears session state to log out the user."""
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.user_role = None
    st.session_state.user_name = None
    st.rerun()

# --- File Extraction Utilities ---

def extract_text_from_file(uploaded_file):
    """Extracts text from PDF, DOCX, or TXT files."""
    file_type = uploaded_file.type
    
    if file_type == "application/pdf":
        try:
            # Using pdfminer.six via imported function
            output_string = io.StringIO()
            uploaded_file.seek(0) # Ensure pointer is at start
            extract_text_to_fp(uploaded_file, output_string)
            return output_string.getvalue()
        except Exception as e:
            return f"Error extracting PDF text: {e}"
            
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            # DOCX handling using python-docx
            doc = Document(uploaded_file)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            return f"Error extracting DOCX text: {e}"

    elif file_type == "text/plain":
        uploaded_file.seek(0)
        return uploaded_file.getvalue().decode("utf-8")
        
    return "Unsupported file format."

# --- NLP & Analysis Core Functions (Unchanged) ---

def clean_text(text):
    """Basic NLP preprocessing: lowercasing, punctuation/number removal."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    return text

def parse_resume(text):
    """Uses spaCy NER and rule-based matching to extract key resume entities."""
    doc = nlp(text)
    
    parsed_data = {
        "name": "N/A", "email": "N/A", "phone": "N/A",
        "skills": [], "education": [], "experience": []
    }
    
    emails = re.findall(r"[a-z0-9\.\-+]+@[a-z0-9\.\-+]+\.[a-z]+", text)
    if emails: parsed_data["email"] = emails[0]
    
    phones = re.findall(r'(?:(?:\+?(\d{1,3}))?[-. (](\d{3})[-. )](\d{3})[-. ]*(\d{4})(?: *[x/#]{1}(\d+))?)|(?:\d{2,4}[-.\s]\d{3,4}[-.\s]\d{3,4})', text)
    if phones: parsed_data["phone"] = "".join(phones[0][:4])

    common_skills = set(["python", "java", "sql", "aws", "azure", "tableau", "react", 
                         "javascript", "html", "css", "machine learning", "deep learning", 
                         "nlp", "pytorch", "tensorflow", "data analysis", "git", "docker"])
    
    found_skills = set()
    for token in doc:
        if token.lemma_.lower() in common_skills:
            found_skills.add(token.lemma_.lower())
    
    for skill in common_skills:
        if skill in text.lower():
            found_skills.add(skill)

    parsed_data["skills"] = list(found_skills)
    
    for ent in doc.ents:
        if ent.label_ == "ORG" and ("university" in ent.text.lower() or "college" in ent.text.lower()):
            if ent.text not in parsed_data["education"]:
                parsed_data["education"].append(ent.text)
        elif ent.label_ in ["ORG", "GPE"] and (ent.text.lower().endswith("inc") or ent.text.lower().endswith("ltd") or "company" in ent.text.lower()):
            if ent.text not in parsed_data["experience"]:
                parsed_data["experience"].append(ent.text)

    return parsed_data

def parse_jd(text):
    """Extracts relevant keywords and required skills from the Job Description."""
    doc = nlp(text)
    
    common_skills = set(["python", "java", "sql", "aws", "azure", "tableau", "react", 
                         "javascript", "html", "css", "machine learning", "deep learning", 
                         "nlp", "pytorch", "tensorflow", "data analysis", "git", "docker"])
    
    required_skills = set()
    for token in doc:
        if token.lemma_.lower() in common_skills:
            required_skills.add(token.lemma_.lower())
            
    for skill in common_skills:
        if skill in text.lower():
            required_skills.add(skill)

    experience_keywords = re.findall(r'(\d+)\s*year(?:s)?\s*(of)?\s*(experience)?', text.lower())
    
    return {
        "required_skills": list(required_skills),
        "experience_keywords": [f"{y} years" for y, of, exp in experience_keywords if y]
    }


def calculate_ats_score(resume_text, jd_text, parsed_resume, parsed_jd):
    """Calculates the ATS Score based on weighted components."""
    
    corpus = [clean_text(resume_text), clean_text(jd_text)]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    keyword_density_score = cosine_sim * 100
    
    resume_skills = set(parsed_resume.get("skills", []))
    required_skills = set(parsed_jd.get("required_skills", []))
    
    if not required_skills:
        skill_match_percent = 100.0
        matched_skills = []
    else:
        matched_skills = list(resume_skills.intersection(required_skills))
        skill_match_percent = (len(matched_skills) / len(required_skills)) * 100
    
    has_experience = 1 if len(parsed_resume.get("experience", [])) > 0 else 0
    experience_match_percent = has_experience * 100
    
    w_skill = 0.50
    w_keyword = 0.30
    w_experience = 0.20
    
    skill_score = min(skill_match_percent, 100.0) * w_skill
    keyword_score = min(keyword_density_score, 100.0) * w_keyword
    experience_score = min(experience_match_percent, 100.0) * w_experience
    
    overall_ats_score = round(skill_score + keyword_score + experience_score, 2)
    
    match_details = {
        "Skill Match": round(min(skill_match_percent, 100.0), 2),
        "Keyword Density": round(min(keyword_density_score, 100.0), 2),
        "Experience Match": round(min(experience_match_percent, 100.0), 2),
        "Matched Skills": matched_skills
    }
    
    return overall_ats_score, match_details, required_skills

def generate_suggestions(required_skills, matched_skills):
    """Generates text suggestions for improvement."""
    missing_skills = required_skills.difference(set(matched_skills))
    
    suggestions = []
    if missing_skills:
        suggestions.append(f"💡 *Missing Key Skills:* Consider adding keywords like *{', '.join(missing_skills)}* to your resume to increase your match score.")
    
    suggestions.append("📄 *ATS-Friendly Templates:* Ensure your resume uses a clean, simple layout (avoid tables, heavy graphics) for better ATS parsing. [Reference Link: Simple Template Guide](https://example.com/ats-guide)")
    suggestions.append("🌟 *Quantify Achievements:* Replace general statements with quantifiable results (e.g., 'Improved performance by 20%').")
    
    return "\n".join(suggestions)

# --- Streamlit UI Components (Dashboards) ---

# Note: The data saving functions are simplified for this demo environment 
# and focus on the current session's analysis. For production, they would 
# implement the user_id filtering for CRUD operations.

def student_dashboard():
    """UI and logic for the Student Dashboard."""
    st.header(f"Student Dashboard: Welcome, {st.session_state.user_name}!")
    st.subheader("ATS Self-Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Upload Your Resume")
        resume_file = st.file_uploader("Upload PDF or DOCX Resume", type=["pdf", "docx", "txt"], key="student_resume")
    
    with col2:
        st.subheader("2. Upload Job Description (Optional)")
        jd_file = st.file_uploader("Upload JD (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"], key="student_jd")
    
    if resume_file:
        resume_text = extract_text_from_file(resume_file)
        
        if "Error" in resume_text:
            st.error(f"Resume extraction failed: {resume_text}")
            return
            
        st.success("Resume uploaded and extracted successfully.")
        
        if jd_file:
            jd_text = extract_text_from_file(jd_file)
            if "Error" in jd_text:
                st.error(f"JD extraction failed: {jd_text}")
                return
        else:
            jd_text = "Highly skilled software engineer with strong Python, Machine Learning, and SQL expertise. Needs 5+ years of experience."
            st.info("No JD uploaded. Using a generic high-demand JD for analysis.")
            
        if st.button("Analyze Resume", type="primary"):
            with st.spinner("Analyzing resume and calculating ATS score..."):
                parsed_resume = parse_resume(resume_text)
                parsed_jd = parse_jd(jd_text)
                
                ats_score, match_details, required_skills = calculate_ats_score(
                    resume_text, jd_text, parsed_resume, parsed_jd
                )
                suggestions = generate_suggestions(set(parsed_jd["required_skills"]), set(match_details["Matched Skills"]))

            # --- Display Results ---
            st.markdown("---")
            st.subheader(f"ATS Score Report: {resume_file.name}")
            
            st.metric("Overall ATS Match Score", f"{ats_score:.2f}%")
            
            st.progress(ats_score / 100.0, text="Match Progress")
            
            st.markdown("### Detailed Match Breakdown")
            
            details_df = pd.DataFrame(match_details.items(), columns=["Metric", "Value"])
            details_df['Value'] = details_df['Value'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else ", ".join(x))
            
            st.dataframe(details_df, use_container_width=True, hide_index=True)

            st.markdown("### Improvement Suggestions")
            st.success(suggestions)
            
            st.markdown("---")
            st.subheader("Parsed Resume Data (For Review)")
            st.json(parsed_resume)
            
def recruiter_dashboard():
    """UI and logic for the Recruiter Dashboard."""
    st.header(f"Recruiter Dashboard: Welcome, {st.session_state.user_name}!")
    
    # 1. JD Upload
    st.subheader("1. Upload Job Description")
    jd_file = st.file_uploader("Upload JD (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"], key="recruiter_jd")
    
    jd_text = ""
    parsed_jd = {}
    if jd_file:
        jd_text = extract_text_from_file(jd_file)
        if "Error" in jd_text:
            st.error(f"JD extraction failed: {jd_text}")
            return
        st.success("JD extracted successfully.")
        parsed_jd = parse_jd(jd_text)
        
        st.info(f"JD Keywords extracted: {', '.join(parsed_jd.get('required_skills', []))}")
        st.markdown("---")
    else:
        st.warning("Please upload a Job Description to proceed.")
        return

    # 2. Resume Bulk Upload
    st.subheader("2. Bulk Upload Candidate Resumes")
    resume_files = st.file_uploader("Upload Resumes (PDF or DOCX)", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="recruiter_resumes")

    if not resume_files:
        st.info("Upload candidate resumes to start the ranking.")
        return
        
    if st.button("Analyze and Rank Candidates", type="primary"):
        results = []
        
        with st.spinner(f"Analyzing {len(resume_files)} resumes against the JD..."):
            required_skills_set = set(parsed_jd["required_skills"])
            
            for resume_file in resume_files:
                try:
                    resume_text = extract_text_from_file(resume_file)
                    if "Error" in resume_text:
                         st.warning(f"Skipping {resume_file.name} due to extraction error.")
                         continue
                         
                    parsed_resume = parse_resume(resume_text)
                    
                    ats_score, match_details, required_skills = calculate_ats_score(
                        resume_text, jd_text, parsed_resume, parsed_jd
                    )
                    
                    missing_skills = required_skills_set.difference(set(match_details["Matched Skills"]))
                    
                    results.append({
                        "Candidate": resume_file.name,
                        "ATS Score (%)": ats_score,
                        "Skill Match (%)": match_details["Skill Match"],
                        "Keyword Density (%)": match_details["Keyword Density"],
                        "Matched Skills": ", ".join(match_details["Matched Skills"]),
                        "Missing Skills": ", ".join(missing_skills) or "None"
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred while processing {resume_file.name}: {e}")
                    
        # 3. Display Results
        if results:
            st.markdown("---")
            st.subheader("Candidate Ranking Results")
            
            df = pd.DataFrame(results)
            df_sorted = df.sort_values(by="ATS Score (%)", ascending=False).reset_index(drop=True)
            df_sorted.index = df_sorted.index + 1
            
            st.dataframe(df_sorted, use_container_width=True)
            
            # Visualization
            st.markdown("### Visualization: Top 5 Candidate ATS Scores")
            top_5 = df_sorted.head(5)
            st.bar_chart(top_5, x="Candidate", y="ATS Score (%)")
            
            # Download link
            csv = df_sorted.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Ranking as CSV",
                data=csv,
                file_name='candidate_ranking.csv',
                mime='text/csv',
            )
        else:
            st.error("No resumes were successfully processed.")

# --- NEW: Login/Signup UI ---

def login_page():
    """Displays the login and signup forms."""
    st.sidebar.title("Access")
    choice = st.sidebar.radio("Go to", ["Login", "Sign Up"])

    if choice == "Login":
        st.subheader("Log in to Resume Analyzer")
        with st.form("login_form"):
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log In", type="primary")

            if submitted:
                if not login_email or not login_password:
                    st.error("Please enter both email and password.")
                    return
                
                auth_result = authenticate_user(login_email, login_password)
                
                if auth_result['logged_in']:
                    st.session_state.logged_in = True
                    st.session_state.user_id = auth_result['user_id']
                    st.session_state.user_role = auth_result['user_role']
                    st.session_state.user_name = auth_result['user_name']
                    st.success(f"Welcome back, {auth_result['user_name']}! Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")
                    
    elif choice == "Sign Up":
        st.subheader("Create a New Account")
        with st.form("signup_form"):
            new_name = st.text_input("Name", key="new_name")
            new_email = st.text_input("Email", key="new_email")
            new_password = st.text_input("Password", type="password", key="new_password")
            new_role = st.selectbox("I am a", ['student', 'recruiter'], key="new_role")
            
            signup_submitted = st.form_submit_button("Sign Up", type="primary")

            if signup_submitted:
                if not all([new_name, new_email, new_password, new_role]):
                    st.error("Please fill in all fields.")
                    return
                    
                success, message = register_user(new_name, new_email, new_password, new_role)
                
                if success:
                    st.success(message)
                    # Switch to login page after successful registration
                    st.session_state.choice = "Login" 
                else:
                    st.error(message)


# --- Main Application Logic ---

def main():
    """Main function to run the Streamlit application, handling auth and routing."""
    st.set_page_config(layout="wide", page_title="Resume Analyzer 🎯")
    
    # Initialize session state for authentication
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.user_role = None
        st.session_state.user_name = None
        
    # 1. Database Setup (must run before any auth or data interaction)
    setup_database()
    
    st.title("🎯 AI Resume & JD Analyzer")
    st.markdown("A tool for students to check ATS-readiness and for recruiters to shortlist candidates.")
    
    if st.session_state.logged_in:
        # User is logged in, show the appropriate dashboard
        st.sidebar.markdown(f"*Logged in as:* {st.session_state.user_name} ({st.session_state.user_role.capitalize()})")
        st.sidebar.button("Logout", on_click=logout)
        st.sidebar.markdown("---")

        if st.session_state.user_role == 'student':
            student_dashboard()
        elif st.session_state.user_role == 'recruiter':
            recruiter_dashboard()
    else:
        # User is not logged in, show login/signup
        login_page()

if __name__ == "__main__":
    main()