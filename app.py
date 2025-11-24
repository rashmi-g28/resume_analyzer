from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
import sqlite3
import pandas as pd
import json
import io
import re
import numpy as np
from datetime import datetime, timedelta
import os

# Advanced NLP Libraries
import spacy
from spacy.matcher import Matcher, PhraseMatcher
import en_core_web_sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
from pdfminer.high_level import extract_text_to_fp
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Download NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except:
    pass

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'jwt-secret-string')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

bcrypt = Bcrypt(app)
jwt = JWTManager(app)
CORS(app)

# Database Configuration
DB_PATH = "resume_analyzer.db"
APP_ID = "resume_analyzer_app"

# Load NLP model
try:
    nlp = en_core_web_sm.load()
except:
    nlp = None

# Enhanced Skill Database
SKILL_DATABASE = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", 
        "swift", "kotlin", "scala", "r", "matlab", "php", "ruby", "perl"
    ],
    "web_technologies": [
        "html", "css", "react", "angular", "vue", "django", "flask", "spring", 
        "express", "node.js", "rest", "graphql", "jquery", "bootstrap"
    ],
    "databases": [
        "sql", "mysql", "postgresql", "mongodb", "redis", "cassandra", "oracle", 
        "sqlite", "dynamodb", "firebase", "elasticsearch"
    ],
    "cloud_technologies": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", 
        "ci/cd", "devops", "ansible", "puppet", "chef"
    ],
    "data_science": [
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", 
        "machine learning", "deep learning", "nlp", "computer vision", 
        "data analysis", "tableau", "power bi", "apache spark", "hadoop"
    ],
    "soft_skills": [
        "leadership", "communication", "teamwork", "problem solving", 
        "critical thinking", "project management", "agile", "scrum"
    ]
}

ALL_SKILLS = [skill for category in SKILL_DATABASE.values() for skill in category]

# Education and Experience Patterns
EDUCATION_PATTERNS = [
    {"label": "EDUCATION", "pattern": [{"LOWER": {"IN": ["university", "college", "institute"]}}]},
    {"label": "DEGREE", "pattern": [{"LOWER": {"IN": ["bachelor", "master", "phd", "mba", "msc", "bsc"]}}]},
]

DEGREE_KEYWORDS = [
    "bachelor", "master", "phd", "doctorate", "mba", "msc", "bsc", "ba", "ma",
    "associate", "diploma", "certificate"
]

# Database Functions
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def setup_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    TABLE_PREFIX = f"{APP_ID}_"

    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}resumes (
        resume_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT,
        text TEXT,
        parsed_json TEXT,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES {TABLE_PREFIX}users(user_id)
    )
    """)
    
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}job_descriptions (
        jd_id INTEGER PRIMARY KEY AUTOINCREMENT,
        recruiter_id INTEGER,
        filename TEXT,
        text TEXT,
        parsed_json TEXT,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(recruiter_id) REFERENCES {TABLE_PREFIX}users(user_id)
    )
    """)
    
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}analysis_results (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        resume_id INTEGER,
        jd_id INTEGER,
        ats_score REAL,
        match_details TEXT,
        suggestions TEXT,
        analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(resume_id) REFERENCES {TABLE_PREFIX}resumes(resume_id),
        FOREIGN KEY(jd_id) REFERENCES {TABLE_PREFIX}job_descriptions(jd_id)
    )
    """)
    
    conn.commit()
    conn.close()

# Authentication Functions
def register_user(name, email, password, role):
    conn = get_db_connection()
    cursor = conn.cursor()
    TABLE_PREFIX = f"{APP_ID}_"
    
    try:
        hashed_pwd = bcrypt.generate_password_hash(password).decode('utf-8')
        cursor.execute(f"""
        INSERT INTO {TABLE_PREFIX}users (name, email, hashed_password, role) 
        VALUES (?, ?, ?, ?)
        """, (name, email, hashed_pwd, role))
        conn.commit()
        user_id = cursor.lastrowid
        return True, "Registration successful", user_id
    except sqlite3.IntegrityError:
        return False, "This email is already registered", None
    except Exception as e:
        return False, f"An unexpected error occurred: {e}", None
    finally:
        conn.close()

def verify_user(email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    TABLE_PREFIX = f"{APP_ID}_"
    
    cursor.execute(f"SELECT user_id, name, hashed_password, role FROM {TABLE_PREFIX}users WHERE email = ?", (email,))
    user_record = cursor.fetchone()
    conn.close()
    
    if user_record and bcrypt.check_password_hash(user_record['hashed_password'], password):
        return {
            'user_id': user_record['user_id'],
            'name': user_record['name'],
            'email': email,
            'role': user_record['role']
        }
    return None

# Enhanced File Processing
def extract_text_from_file(file_content, filename):
    """Extract text from file bytes based on file type."""
    file_extension = filename.split('.')[-1].lower()
    
    try:
        if file_extension == 'pdf':
            output_string = io.StringIO()
            file_stream = io.BytesIO(file_content)
            extract_text_to_fp(file_stream, output_string)
            text = output_string.getvalue()
            return clean_extracted_text(text)
            
        elif file_extension == 'docx':
            file_stream = io.BytesIO(file_content)
            doc = Document(file_stream)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return clean_extracted_text(text)
            
        elif file_extension == 'txt':
            return clean_extracted_text(file_content.decode('utf-8'))
            
        else:
            return None, "Unsupported file format"
    except Exception as e:
        return None, f"Error extracting text: {str(e)}"

def clean_extracted_text(text):
    """Clean and normalize extracted text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.@\+\-\(\)\:\,]', '', text)
    return text.strip()

# Advanced NLP Functions
def extract_personal_info(text):
    """Extract personal information using multiple strategies."""
    info = {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "location": extract_location(text)
    }
    return info

def extract_name(text):
    """Enhanced name extraction."""
    # Method 1: Look for email-based names
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if emails:
        name_from_email = emails[0].split('@')[0]
        name_from_email = re.sub(r'[0-9_]', ' ', name_from_email)
        name_parts = [part.capitalize() for part in name_from_email.split('.') if len(part) > 1]
        if name_parts:
            return ' '.join(name_parts)
    
    # Method 2: Use spaCy NER
    if nlp:
        doc = nlp(text[:2000])
        for ent in doc.ents:
            if ent.label_ == "PERSON" and 2 <= len(ent.text.split()) <= 4:
                return ent.text.title()
    
    # Method 3: Pattern-based extraction
    lines = text.split('\n')
    for line in lines[:10]:
        line = line.strip()
        if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', line):
            return line
    
    return "Not Found"

def extract_email(text):
    """Extract email addresses."""
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return emails[0] if emails else "Not Found"

def extract_phone(text):
    """Extract phone numbers."""
    phone_patterns = [
        r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
    ]
    
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        if phones:
            phone = phones[0] if isinstance(phones[0], str) else ''.join(phones[0])
            return re.sub(r'[^\d+]', '', phone)
    
    return "Not Found"

def extract_location(text):
    """Extract location information."""
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "GPE":
                return ent.text
    return "Not Found"

def extract_skills_advanced(text):
    """Advanced skill extraction using multiple techniques."""
    skills_found = set()
    
    # Direct keyword matching
    for skill in ALL_SKILLS:
        if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
            skills_found.add(skill)
    
    # NLP-based extraction
    if nlp:
        doc = nlp(text.lower())
        
        # Check noun phrases for skills
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            for skill in ALL_SKILLS:
                if skill in chunk_text and skill not in skills_found:
                    skills_found.add(skill)
        
        # Check entities
        for ent in doc.ents:
            if ent.text.lower() in ALL_SKILLS:
                skills_found.add(ent.text.lower())
    
    return list(skills_found)

def extract_education(text):
    """Extract education information."""
    education_entries = []
    
    # Degree patterns
    for degree in DEGREE_KEYWORDS:
        pattern = rf'\b{degree}[^.,]*(?:in|of)?[^.,]*'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            education_entries.append(match.group().strip())
    
    # University/College patterns
    institution_pattern = r'([A-Z][a-zA-Z\s]+(?:University|College|Institute|School))'
    institutions = re.findall(institution_pattern, text)
    education_entries.extend(institutions)
    
    # Education section extraction
    education_section = re.search(r'(?:education|academic)[^a-z]*?(?:\n|\.)(.*?)(?:\n\n|\n[A-Z]|$)', 
                                 text, re.IGNORECASE | re.DOTALL)
    if education_section:
        lines = education_section.group(1).split('\n')
        for line in lines:
            if len(line.strip()) > 10:
                education_entries.append(line.strip())
    
    return list(set(entry for entry in education_entries if len(entry) > 5))[:10]

def extract_experience(text):
    """Extract work experience information."""
    experience_entries = []
    
    # Experience duration patterns
    duration_patterns = [
        r'(\d+[\+\-\–]?\d*)\s*(?:year|yr)s?\s*(?:of)?\s*(?:experience)',
        r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}\s*(?:to|–|-)\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}',
        r'(?:present|current|now)'
    ]
    
    for pattern in duration_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            experience_entries.append(match.group())
    
    # Job title patterns
    job_titles = [
        r'\b(?:software\s+engineer|developer|programmer)\b',
        r'\b(?:data\s+scientist|machine\s+learning\s+engineer)\b',
        r'\b(?:web\s+developer|frontend|backend|full\s+stack)\b',
        r'\b(?:devops\s+engineer|cloud\s+engineer)\b',
        r'\b(?:product\s+manager|project\s+manager)\b'
    ]
    
    for pattern in job_titles:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Get context around job title
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 100)
            context = text[start:end].replace('\n', ' ')
            experience_entries.append(context.strip())
    
    return list(set(experience_entries))[:15]

def parse_resume_advanced(text):
    """Advanced resume parsing with comprehensive extraction."""
    if nlp is None:
        return {"error": "NLP model not available"}
    
    parsed_data = {
        "personal_info": extract_personal_info(text),
        "skills": extract_skills_advanced(text),
        "education": extract_education(text),
        "experience": extract_experience(text),
        "summary": extract_summary(text)
    }
    
    return parsed_data

def extract_summary(text):
    """Extract a summary from the resume."""
    sentences = sent_tokenize(text)
    if sentences:
        # Return first meaningful sentence (not too short, not too long)
        for sentence in sentences:
            if 20 < len(sentence) < 200:
                return sentence.strip()
        return sentences[0][:150] + "..." if len(sentences[0]) > 150 else sentences[0]
    return "No summary available"

def parse_job_description(text):
    """Advanced job description parsing."""
    parsed_data = {
        "required_skills": extract_skills_advanced(text),
        "experience_requirements": extract_experience_requirements(text),
        "education_requirements": extract_education_requirements(text),
        "keywords": extract_keywords(text)
    }
    
    return parsed_data

def extract_experience_requirements(text):
    """Extract experience requirements from JD."""
    experience_matches = re.findall(r'(\d+[\+\-\–]?\d*)\s*(?:year|yr)s?\s*(?:of)?\s*(?:experience)', text, re.IGNORECASE)
    return [f"{exp} years experience" for exp in experience_matches]

def extract_education_requirements(text):
    """Extract education requirements from JD."""
    education_matches = re.findall(r'(bachelor|master|phd|mba|msc|bsc|ba|ma|degree|diploma)[^\.,]*(?:in|of)?[^\.\,]*', text, re.IGNORECASE)
    return education_matches

def extract_keywords(text):
    """Extract important keywords from text."""
    if nlp is None:
        return []
    
    doc = nlp(text.lower())
    keywords = [
        token.lemma_ for token in doc 
        if token.pos_ in ['NOUN', 'ADJ'] 
        and len(token.lemma_) > 3 
        and token.lemma_ not in stopwords.words('english')
    ]
    
    return list(set(keywords))[:25]

# Analysis Functions
def calculate_advanced_ats_score(resume_text, jd_text, parsed_resume, parsed_jd):
    """Calculate comprehensive ATS score with multiple factors."""
    
    # 1. Keyword Similarity
    keyword_score = calculate_keyword_similarity(resume_text, jd_text)
    
    # 2. Skill Matching
    skill_score, matched_skills, missing_skills = calculate_skill_match(
        parsed_resume.get("skills", []),
        parsed_jd.get("required_skills", [])
    )
    
    # 3. Experience Matching
    experience_score = calculate_experience_match(
        parsed_resume.get("experience", []),
        parsed_jd.get("experience_requirements", [])
    )
    
    # 4. Education Matching
    education_score = calculate_education_match(
        parsed_resume.get("education", []),
        parsed_jd.get("education_requirements", [])
    )
    
    # Weighted overall score
    weights = {
        "skill": 0.40,
        "keyword": 0.30,
        "experience": 0.20,
        "education": 0.10
    }
    
    overall_score = (
        skill_score * weights["skill"] +
        keyword_score * weights["keyword"] +
        experience_score * weights["experience"] +
        education_score * weights["education"]
    )
    
    match_details = {
        "skill_match": round(skill_score, 2),
        "keyword_density": round(keyword_score, 2),
        "experience_match": round(experience_score, 2),
        "education_match": round(education_score, 2),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }
    
    return round(overall_score, 2), match_details

def calculate_keyword_similarity(text1, text2):
    """Calculate keyword similarity using TF-IDF and cosine similarity."""
    try:
        corpus = [clean_text_for_analysis(text1), clean_text_for_analysis(text2)]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return min(cosine_sim * 100, 100)
    except:
        return 0

def clean_text_for_analysis(text):
    """Clean text for analysis."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def calculate_skill_match(resume_skills, required_skills):
    """Calculate skill matching percentage."""
    resume_skills_set = set(resume_skills)
    required_skills_set = set(required_skills)
    
    if not required_skills_set:
        return 100.0, [], []
    
    matched_skills = list(resume_skills_set.intersection(required_skills_set))
    missing_skills = list(required_skills_set - resume_skills_set)
    
    match_percentage = (len(matched_skills) / len(required_skills_set)) * 100
    return min(match_percentage, 100), matched_skills, missing_skills

def calculate_experience_match(resume_experience, jd_requirements):
    """Calculate experience matching."""
    if not jd_requirements:
        return 100.0
    
    # Simple check: if resume has experience entries and JD requires experience
    has_experience = len(resume_experience) > 0
    jd_requires_experience = len(jd_requirements) > 0
    
    if jd_requires_experience and has_experience:
        return 100.0
    elif not jd_requires_experience:
        return 100.0
    else:
        return 0.0

def calculate_education_match(resume_education, jd_requirements):
    """Calculate education matching."""
    if not jd_requirements:
        return 100.0
    
    # Simple check for education presence
    has_education = len(resume_education) > 0
    if has_education:
        return 100.0
    else:
        return 0.0

def generate_improvement_suggestions(match_details):
    """Generate personalized improvement suggestions."""
    suggestions = []
    
    if match_details["skill_match"] < 70:
        missing_skills = match_details.get("missing_skills", [])
        if missing_skills:
            suggestions.append({
                "type": "skills",
                "priority": "high",
                "message": f"Add these missing skills to your resume: {', '.join(missing_skills[:5])}",
                "action": "Consider gaining experience with these technologies or highlighting relevant projects."
            })
    
    if match_details["keyword_density"] < 60:
        suggestions.append({
            "type": "keywords",
            "priority": "medium",
            "message": "Increase keyword density by incorporating more relevant terms from the job description.",
            "action": "Mirror the language used in the job description throughout your resume."
        })
    
    if match_details["experience_match"] < 100:
        suggestions.append({
            "type": "experience",
            "priority": "medium",
            "message": "Highlight relevant work experience and quantify achievements.",
            "action": "Use numbers to demonstrate impact (e.g., 'Improved performance by 25%')."
        })
    
    # General suggestions
    general_suggestions = [
        {
            "type": "format",
            "priority": "low",
            "message": "Use an ATS-friendly format with clear section headings.",
            "action": "Avoid tables, columns, and graphics that might not parse correctly."
        },
        {
            "type": "customization",
            "priority": "medium",
            "message": "Tailor your resume for each specific job application.",
            "action": "Customize your resume to match the specific requirements of each job description."
        }
    ]
    
    suggestions.extend(general_suggestions)
    return suggestions

# API Routes
@app.route('/api/register', methods=['POST'])
def register():
    """User registration endpoint."""
    data = request.get_json()
    
    if not data or not all(k in data for k in ['name', 'email', 'password', 'role']):
        return jsonify({"error": "Missing required fields"}), 400
    
    success, message, user_id = register_user(
        data['name'], 
        data['email'], 
        data['password'], 
        data['role']
    )
    
    if success:
        return jsonify({
            "message": message,
            "user_id": user_id
        }), 201
    else:
        return jsonify({"error": message}), 400

@app.route('/api/login', methods=['POST'])
def login():
    """User login endpoint."""
    data = request.get_json()
    
    if not data or not all(k in data for k in ['email', 'password']):
        return jsonify({"error": "Email and password required"}), 400
    
    user = verify_user(data['email'], data['password'])
    
    if user:
        access_token = create_access_token(identity=user['user_id'])
        return jsonify({
            "message": "Login successful",
            "access_token": access_token,
            "user": user
        }), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401

@app.route('/api/analyze/resume', methods=['POST'])
@jwt_required()
def analyze_resume():
    """Analyze a single resume with optional job description."""
    user_id = get_jwt_identity()
    
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file provided"}), 400
    
    resume_file = request.files['resume']
    jd_file = request.files.get('job_description')
    
    if resume_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Extract text from resume
    resume_content = resume_file.read()
    resume_text, error = extract_text_from_file(resume_content, resume_file.filename)
    
    if error:
        return jsonify({"error": error}), 400
    
    # Extract text from job description if provided
    jd_text = None
    if jd_file and jd_file.filename != '':
        jd_content = jd_file.read()
        jd_text, jd_error = extract_text_from_file(jd_content, jd_file.filename)
        if jd_error:
            return jsonify({"error": jd_error}), 400
    else:
        # Use default JD for analysis
        jd_text = """
        We are seeking a skilled professional with experience in software development, 
        data analysis, and cloud technologies. The ideal candidate should have strong 
        problem-solving skills, proficiency in programming languages like Python or Java, 
        and experience with modern development practices. A bachelor's degree in 
        Computer Science or related field is preferred.
        """
    
    # Parse documents
    parsed_resume = parse_resume_advanced(resume_text)
    parsed_jd = parse_job_description(jd_text)
    
    # Calculate ATS score
    ats_score, match_details = calculate_advanced_ats_score(
        resume_text, jd_text, parsed_resume, parsed_jd
    )
    
    # Generate suggestions
    suggestions = generate_improvement_suggestions(match_details)
    
    # Save to database (optional)
    save_analysis_result(user_id, resume_file.filename, jd_file.filename if jd_file else "default", 
                        ats_score, match_details, suggestions)
    
    return jsonify({
        "success": True,
        "resume_analysis": parsed_resume,
        "job_description_analysis": parsed_jd,
        "ats_score": ats_score,
        "match_details": match_details,
        "suggestions": suggestions
    }), 200

@app.route('/api/analyze/bulk', methods=['POST'])
@jwt_required()
def analyze_bulk_resumes():
    """Analyze multiple resumes against a job description."""
    user_id = get_jwt_identity()
    
    if 'job_description' not in request.files:
        return jsonify({"error": "No job description provided"}), 400
    
    jd_file = request.files['job_description']
    resume_files = request.files.getlist('resumes')
    
    if not resume_files:
        return jsonify({"error": "No resume files provided"}), 400
    
    # Extract JD text
    jd_content = jd_file.read()
    jd_text, jd_error = extract_text_from_file(jd_content, jd_file.filename)
    
    if jd_error:
        return jsonify({"error": jd_error}), 400
    
    parsed_jd = parse_job_description(jd_text)
    results = []
    
    for resume_file in resume_files:
        if resume_file.filename == '':
            continue
            
        try:
            resume_content = resume_file.read()
            resume_text, resume_error = extract_text_from_file(resume_content, resume_file.filename)
            
            if resume_error:
                results.append({
                    "filename": resume_file.filename,
                    "error": resume_error,
                    "ats_score": 0
                })
                continue
            
            parsed_resume = parse_resume_advanced(resume_text)
            ats_score, match_details = calculate_advanced_ats_score(
                resume_text, jd_text, parsed_resume, parsed_jd
            )
            
            results.append({
                "filename": resume_file.filename,
                "candidate_name": parsed_resume["personal_info"]["name"],
                "email": parsed_resume["personal_info"]["email"],
                "ats_score": ats_score,
                "skill_match": match_details["skill_match"],
                "keyword_density": match_details["keyword_density"],
                "matched_skills_count": len(match_details["matched_skills"]),
                "missing_skills_count": len(match_details["missing_skills"]),
                "matched_skills": match_details["matched_skills"][:5],
                "missing_skills": match_details["missing_skills"][:3]
            })
            
        except Exception as e:
            results.append({
                "filename": resume_file.filename,
                "error": str(e),
                "ats_score": 0
            })
    
    # Sort by ATS score
    results.sort(key=lambda x: x.get('ats_score', 0), reverse=True)
    
    return jsonify({
        "success": True,
        "job_description": parsed_jd,
        "candidates_ranking": results,
        "total_analyzed": len([r for r in results if 'error' not in r])
    }), 200

def save_analysis_result(user_id, resume_filename, jd_filename, ats_score, match_details, suggestions):
    """Save analysis result to database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        TABLE_PREFIX = f"{APP_ID}_"
        
        # Save resume
        cursor.execute(f"""
        INSERT INTO {TABLE_PREFIX}resumes (user_id, filename, text, parsed_json)
        VALUES (?, ?, ?, ?)
        """, (user_id, resume_filename, "", json.dumps(match_details)))
        
        resume_id = cursor.lastrowid
        
        # Save JD (using a placeholder since we don't store the actual JD text)
        cursor.execute(f"""
        INSERT INTO {TABLE_PREFIX}job_descriptions (recruiter_id, filename, text, parsed_json)
        VALUES (?, ?, ?, ?)
        """, (user_id, jd_filename, "", json.dumps({"type": "analysis_jd"})))
        
        jd_id = cursor.lastrowid
        
        # Save result
        cursor.execute(f"""
        INSERT INTO {TABLE_PREFIX}analysis_results 
        (resume_id, jd_id, ats_score, match_details, suggestions)
        VALUES (?, ?, ?, ?, ?)
        """, (resume_id, jd_id, ats_score, json.dumps(match_details), json.dumps(suggestions)))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Error saving analysis result: {e}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "nlp_loaded": nlp is not None
    }), 200

# Initialize database
@app.before_first_request
def initialize():
    setup_database()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
