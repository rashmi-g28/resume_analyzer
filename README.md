🎯 AI Resume & JD Analyzer

This is a full-stack, single-file Streamlit application designed to analyze resumes against job descriptions (JDs) using NLP techniques. This version includes full user authentication using SQLite and bcrypt.

⚙ Prerequisites

Python 3.8+

Required Libraries: Install the dependencies listed in requirements.txt.

spaCy Model: You must download the necessary English model for spaCy.

🛠 Local Setup Instructions

Clone the Repository / Save the Files:
Save app.py and requirements.txt into a new folder (e.g., resume_analyzer/).

Install Python Dependencies:
Open your terminal/command prompt, navigate to the folder where you saved the files, and run:

pip install -r requirements.txt


(This step now includes the bcrypt package for secure password hashing.)

Download the spaCy Model:
The application requires the small English language model (en_core_web_sm). Run the following command:

python -m spacy download en_core_web_sm


Run the Streamlit Application:
Start the application from your terminal:

streamlit run app.py


This will open the application in your web browser, typically at http://localhost:8501.

🔒 Authentication and Access Flow

Upon running the application for the first time, it will initialize the SQLite database (resume_analyzer.db) and create the users table.

Sign Up: Navigate to the Sign Up tab in the sidebar.

Enter a Name, Email, and Password.

Select your Role (student or recruiter).

Click "Sign Up". Your password will be securely hashed using bcrypt.

Log In: Navigate to the Login tab.

Enter the registered Email and Password.

Click "Log In".

Role-Based Access: After successful login, the application automatically redirects you to the correct dashboard:

If Role = student, you see the Student Dashboard.

If Role = recruiter, you see the Recruiter Dashboard.

Data Segregation: The underlying database structure is now ready to link all uploaded resumes, JDs, and results to the specific user_id of the logged-in user, ensuring secure data separation (though the full data persistence implementation is simplified for this demo environment).

Logout: A Logout button is displayed in the sidebar to clear the session and return you to the Login screen.

📝 Database Schema (Updated)

The application automatically creates an SQLite database file (resume_analyzer.db) with the following updated tables:

{APP_ID}_users: Stores user credentials (user_id, email, hashed_password, role).

{APP_ID}_resumes: Stores uploaded resume data, linked via user_id.

{APP_ID}_job_descriptions: Stores uploaded JD data, linked via recruiter_id.

{APP_ID}_results: Stores the ATS score and analysis results.