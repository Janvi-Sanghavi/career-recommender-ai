import os
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent sklearn/scipy thread hang on Windows

from flask import Flask, render_template, request, jsonify, session
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
from serpapi.google_search import GoogleSearch # ✅ SERP API import

app = Flask(__name__)
app.secret_key = 'dummy_key_for_dev_only'

# -------------------------------
# API Keys
# -------------------------------
SERP_API_KEY = "e77f4c05a867cc5ef39855ee77448e2bb35572a8c95921631f0e43007b1a9526"  # 🔑 Replace with your real key
GOOGLE_LLM_API_KEY = "AIzaSyD1fBfUSEYZG2Y4T5_sW3jj4xVN2A-I78E"

# -------------------------------
# Initialize Google Gemini Model
# -------------------------------
try:
    google_client = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_LLM_API_KEY)
except Exception as e:
    print(f"Google LLM setup failed: {e}")
    google_client = None

# -------------------------------
# Sample Training Data
# -------------------------------
student_df = pd.DataFrame({
    'interests': ['coding tech', 'art design', 'business marketing', 'teaching education',
                  'entrepreneurship startup', 'data analysis stats', 'ai machine learning', 'coding web development'],
    'skills': ['python java', 'photoshop illustrator', 'excel seo', 'communication planning',
               'leadership pitching', 'sql pandas', 'tensorflow pytorch', 'html css js react'],
    'stream': ['Science', 'Arts', 'Commerce', 'Arts', 'Commerce', 'Science', 'Science', 'Science'],
    'grade_level': ['Graduate', '12th', 'Graduate', '10th', 'Graduate', '12th', 'Graduate', '12th'],
    'recommended_career': ['Software Engineer', 'Graphic Designer', 'Marketing Specialist', 'Teacher',
                           'Entrepreneur', 'Data Scientist', 'AI/ML Engineer', 'Software Engineer'],
    'recommended_job': ['Backend Developer', 'UI/UX Designer', 'Marketing Analyst', 'Teacher',
                        'Startup Founder', 'Data Analyst', 'DevOps Engineer', 'Frontend Developer'],
    'recommended_course': ['M.Tech Computer Science', 'Diploma in Graphic Design', 'MBA Marketing', 'B.Ed Education',
                           'Postgraduate Diploma in Entrepreneurship', 'M.Sc Data Science', 'Advanced Diploma in AI/ML',
                           'B.Tech Software Engineering']
})

profiles = (student_df['interests'] + ' ' + student_df['skills'] + ' ' +
            student_df['stream'] + ' ' + student_df['grade_level']).tolist()

# -------------------------------
# Train Models
# -------------------------------
def train_model():
    def train_for(target_col):
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(profiles)
        le = LabelEncoder()
        y = le.fit_transform(student_df[target_col])
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        return vectorizer, le, clf
    return train_for('recommended_career'), train_for('recommended_job'), train_for('recommended_course')

(career_vectorizer, le_career, clf_career), (job_vectorizer, le_job, clf_job), (course_vectorizer, le_course, clf_course) = train_model()

# -------------------------------
# SERP API Fetch Functions
# -------------------------------
def fetch_colleges(location, stream=None, course=None):
    """
    Fetches top commerce/science/arts college names using SERP API.
    Returns a clean list of up to 5 college names with better accuracy.
    """
    if course:
        query = f"top {stream or ''} colleges in {location} India offering {course}"
    elif stream:
        query = f"top {stream} colleges in {location} India 2025 NIRF ranking"
    else:
        query = f"top colleges in {location} India 2025 NIRF ranking"

    try:
        params = {"engine": "google", "q": query, "api_key": SERP_API_KEY}
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])

        college_names = []
        keywords = ["college", "institute", "university", "academy", "school"]

        for item in organic_results[:10]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            combined = f"{title} {snippet}"

            # Improved pattern: capture full proper names like "Narsee Monjee College of Commerce and Economics"
            matches = re.findall(
                r"([A-Z][A-Za-z&.\s,'-]*(?:University|College|Institute|Academy|School)(?: of [A-Z][A-Za-z\s,&'-]*)?)",
                combined
            )
            for m in matches:
                clean_name = re.sub(r"\s+", " ", m).strip()
                if any(k in clean_name.lower() for k in keywords):
                    if clean_name not in college_names:
                        college_names.append(clean_name)

        # If SERP API gives valid names
        if college_names:
            return college_names[:5]

    except Exception as e:
        print(f"College fetch error: {e}")

    # --- Fallbacks (Commerce Focus) ---
    fallback = {
        "mumbai": [
            "Narsee Monjee College of Commerce and Economics (NM College)",
            "H.R. College of Commerce and Economics",
            "St. Xavier’s College, Mumbai",
            "Mithibai College of Arts, Science & Commerce",
            "K.P.B. Hinduja College of Commerce"
        ],
        "ahmedabad": [
            "HL College of Commerce",
            "Ahmedabad University",
            "GLS University",
            "Nirma University",
            "St. Xavier’s College, Ahmedabad"
        ],
        "delhi": [
            "Shri Ram College of Commerce (SRCC)",
            "Hansraj College, Delhi University",
            "Kirori Mal College, DU",
            "Lady Shri Ram College for Women (LSR)",
            "Jesus and Mary College"
        ],
        "gujarat": [
            "MS University, Vadodara",
            "PDPU Gandhinagar",
            "Nirma University",
            "GLS University, Ahmedabad",
            "Parul University, Vadodara"
        ],
        "india": [
            "Shri Ram College of Commerce (Delhi University)",
            "Christ University (Bangalore)",
            "Loyola College (Chennai)",
            "Symbiosis College of Arts and Commerce (Pune)",
            "Jain (Deemed-to-be University), Bangalore"
        ]
    }

    loc = location.lower()
    for key in fallback:
        if key in loc:
            return fallback[key]

    # Generic backup
    return [
        "Christ University (Bangalore)",
        "Narsee Monjee College (Mumbai)",
        "SRCC (Delhi University)",
        "Symbiosis College (Pune)",
        "Loyola College (Chennai)"
    ]


def fetch_jobs(location, skills=None):
    """
    Fetches top job roles or openings for a given location and skill area.
    Filters out article titles and keeps only realistic job names.
    """
    if not skills or skills.strip().lower() in ['none', '']:
        skills = 'tech'
    query = f"top {skills} job openings in {location} India 2025"

    try:
        params = {"engine": "google", "q": query, "api_key": SERP_API_KEY}
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        job_names = []

        exclude_words = ["Top", "Best", "Career", "Guide", "Article", "Courses", "2025", "Jobs in India"]

        for item in organic_results[:10]:
            title = item.get("title", "")
            if any(word in title for word in exclude_words):
                continue
            # Extract short job titles like "Data Analyst", "Software Engineer", etc.
            matches = re.findall(r"[A-Z][A-Za-z\s/&-]*(Engineer|Developer|Analyst|Designer|Manager|Scientist|Intern|Specialist)", title)
            for m in matches:
                clean_job = m.strip()
                if clean_job not in job_names:
                    job_names.append(clean_job)

        if job_names:
            return job_names[:5]

    except Exception as e:
        print(f"Job fetch error: {e}")

    # Fallback realistic job lists
    fallback = {
        "mumbai": ["Software Engineer – TCS", "Frontend Developer – Accenture", "Data Analyst – Infosys", "Digital Marketing Executive – Byju’s", "AI Intern – Zycus"],
        "ahmedabad": ["Python Developer – eInfochips", "Web Developer – Tata Consultancy", "Data Analyst – Adani", "Support Engineer – Wipro", "Intern – Zydus"],
        "gujarat": ["Backend Developer – Reliance Jio", "Data Analyst – HDFC", "Frontend Engineer – TATA", "AI Intern – IIT Gandhinagar", "Tester – Wipro"],
        "india": ["Software Engineer – Google India", "Full-Stack Developer – Microsoft", "Data Scientist – Amazon", "DevOps Engineer – Flipkart", "App Developer – Paytm"]
    }

    loc = location.lower()
    for key in fallback:
        if key in loc:
            return fallback[key]

    return ["Job 1", "Job 2", "Job 3", "Job 4", "Job 5"]

# -------------------------------
# Flask Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/init_chat", methods=["POST"])
def init_chat():
    name = request.form.get("name", "")
    interests = request.form.get("interests", "")
    skills = request.form.get("skills", "")
    stream_input = request.form.get("stream", "")
    grade_level = request.form.get("grade_level", "")
    location = request.form.get("location", "")

    if not all([name, interests, skills, stream_input, grade_level, location]):
        return jsonify({'error': 'All fields are required'}), 400

    session['profile'] = {
        'name': name,
        'interests': interests,
        'skills': skills,
        'stream': stream_input,
        'grade_level': grade_level,
        'location': location
    }
    session['chat_history'] = []

    user_profile = f"{interests} {skills} {stream_input} {grade_level}"

    # Predictions
    user_vec_career = career_vectorizer.transform([user_profile])
    recommended_careers = [le_career.classes_[i] for i in np.argsort(clf_career.predict_proba(user_vec_career)[0])[-3:][::-1]]
    user_vec_job = job_vectorizer.transform([user_profile])
    recommended_jobs = [le_job.classes_[i] for i in np.argsort(clf_job.predict_proba(user_vec_job)[0])[-3:][::-1]]
    user_vec_course = course_vectorizer.transform([user_profile])
    recommended_courses = [le_course.classes_[i] for i in np.argsort(clf_course.predict_proba(user_vec_course)[0])[-3:][::-1]]

    local_list = ""
    india_list = ""
    if grade_level.lower() in ['10th', '12th']:
        top_course = recommended_courses[0] if recommended_courses else stream_input
        local_colleges = fetch_colleges(location, stream_input, top_course)
        top_colleges = fetch_colleges("India", stream_input, top_course)
        local_list = "\n- ".join(local_colleges) if local_colleges else ""
        india_list = "\n- ".join(top_colleges) if top_colleges else ""
    else:
        # Graduate students — suggest job openings and postgraduate colleges
        local_jobs = fetch_jobs(location, skills)
        top_jobs = fetch_jobs("India", skills)
        local_list = "\n- ".join(local_jobs) if local_jobs else ""
        india_list = "\n- ".join(top_jobs) if top_jobs else ""

        # 🔹 Fetch postgraduate colleges (based on top recommended course)
        top_course = recommended_courses[0] if recommended_courses else "Postgraduate studies"
        pg_colleges_local = fetch_colleges(location, stream_input, top_course)
        pg_colleges_india = fetch_colleges("India", stream_input, top_course)
        pg_local_list = "\n- ".join(pg_colleges_local) if pg_colleges_local else ""
        pg_india_list = "\n- ".join(pg_colleges_india) if pg_colleges_india else ""

        # 🔹 Skill recommendations
        skill_suggestions = {
            "Software Engineer": ["DSA", "System Design", "Cloud Computing (AWS/Azure)", "Docker & GitHub"],
            "Data Scientist": ["SQL", "Machine Learning", "Python", "Power BI/Tableau"],
            "AI/ML Engineer": ["TensorFlow", "Deep Learning", "MLOps", "Prompt Engineering"],
            "Marketing Specialist": ["SEO", "Google Analytics", "Copywriting", "Digital Marketing"],
            "Entrepreneur": ["Pitch Deck Creation", "Finance Basics", "Networking", "Growth Hacking"],
            "Teacher": ["Educational Psychology", "Online Teaching Tools", "Curriculum Design"],
            "Graphic Designer": ["Figma", "Canva", "Motion Graphics", "3D Design"]
        }
        recommended_main = recommended_careers[0] if recommended_careers else "Professional"
        suggested_skills = skill_suggestions.get(recommended_main, ["Advanced Communication", "Project Management", "Leadership"])

    # ----------------------------------------
    # 💡 Enhanced Human-like AI Prompt
    # ----------------------------------------
    if grade_level.lower() in ['10th', '12th']:
        chat_prompt = f"""
You are CareerBot 🤖 — a friendly and professional career counsellor.

User Profile:
- Name: {name}
- Interests: {interests}
- Skills: {skills}
- Stream: {stream_input}
- Grade Level: {grade_level}
- Location: {location}

System Recommendations:
- Careers: {', '.join(recommended_careers)}
- Courses: {', '.join(recommended_courses)}
- Jobs: {', '.join(recommended_jobs)}

🎓 Top Colleges in {location}:
- {local_list}

🏛️ Top Colleges in India:
- {india_list}
"""
    else:
        chat_prompt = f"""
You are CareerBot 🤖 — a smart and encouraging mentor for graduates.

User Profile:
- Name: {name}
- Interests: {interests}
- Skills: {skills}
- Stream: {stream_input}
- Grade Level: {grade_level}
- Location: {location}

System Recommendations:
- Careers: {', '.join(recommended_careers)}
- Courses: {', '.join(recommended_courses)}
- Jobs: {', '.join(recommended_jobs)}

💼 Job Openings in {location}:
- {local_list}

🌐 Job Openings in India:
- {india_list}

🎓 Top Postgraduate Colleges in {location}:
- {pg_local_list}

🏛️ Top Postgraduate Colleges in India:
- {pg_india_list}

💡 Skills to Gain Next:
- {', '.join(suggested_skills)}
"""

    # Generate Gemini response
    response_text = ""
    try:
        if google_client:
            response = google_client.invoke(chat_prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
        else:
            raise Exception("LLM not active")
    except Exception as e:
        print(f"LLM invoke error: {e}")
        response_text = f"""### CareerBot’s Top Recommendation for {name}:

Based on your profile, you could excel in **{recommended_careers[0]}**.

🎯 Recommended Courses: {', '.join(recommended_courses)}  
💼 Recommended Jobs: {', '.join(recommended_jobs)}  

💡 Keep learning, stay curious, and you’ll succeed! 🌟"""

    session['chat_history'].append({'role': 'bot', 'content': response_text})
    session.modified = True
    return jsonify({'response': response_text, 'chat_history': session['chat_history']})


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get('message', '')
    profile = session.get('profile', {})
    chat_history = session.get('chat_history', [])
    history_str = '\n'.join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])

    name = profile.get('name', 'User')
    stream_input = profile.get('stream', '')
    grade_level = profile.get('grade_level', '')
    interests = profile.get('interests', '')
    skills = profile.get('skills', '')

    user_profile = f"{interests} {skills} {stream_input} {grade_level}"
    user_vec_career = career_vectorizer.transform([user_profile])
    recommended_careers = [le_career.classes_[i] for i in np.argsort(clf_career.predict_proba(user_vec_career)[0])[-3:][::-1]]
    recommended_main = recommended_careers[0] if recommended_careers else "Professional"

    skill_suggestions = {
        "Software Engineer": ["DSA", "System Design", "Cloud Computing (AWS/Azure)", "Docker & GitHub"],
        "Data Scientist": ["SQL", "Machine Learning", "Python", "Power BI/Tableau"],
        "AI/ML Engineer": ["TensorFlow", "Deep Learning", "MLOps", "Prompt Engineering"],
        "Marketing Specialist": ["SEO", "Google Analytics", "Copywriting", "Digital Marketing"],
        "Entrepreneur": ["Pitch Deck Creation", "Finance Basics", "Networking", "Growth Hacking"],
        "Teacher": ["Educational Psychology", "Online Teaching Tools", "Curriculum Design"],
        "Graphic Designer": ["Figma", "Canva", "Motion Graphics", "3D Design"]
    }
    suggested_skills = skill_suggestions.get(recommended_main, ["Advanced Communication", "Project Management", "Leadership"])

    skill_keywords = ["skill", "learn", "improve", "course", "next step", "upskill", "develop"]
    is_skill_query = any(keyword in user_message.lower() for keyword in skill_keywords)

    if is_skill_query:
        response_text = f"""
### 💡 Skills to Gain Next for {recommended_main}:
To become stronger in your field, focus on:
- {', '.join(suggested_skills)}

Each of these will help you grow professionally. 🌱
"""
    else:
        chat_prompt = f"""
Previous conversation:
{history_str}

User profile: {profile}
User message: {user_message}

Reply as CareerBot 🤖 — supportive, helpful, and encouraging.
"""
        response_text = ""
        try:
            if google_client:
                response = google_client.invoke(chat_prompt)
                response_text = response.content if hasattr(response, "content") else str(response)
            else:
                raise Exception("LLM not active")
        except Exception as e:
            print(f"Chat error: {e}")
            response_text = "That’s great! Keep exploring opportunities that match your interests."

    chat_history.append({'role': 'user', 'content': user_message})
    chat_history.append({'role': 'bot', 'content': response_text})
    session['chat_history'] = chat_history[-10:]
    session.modified = True
    return jsonify({'response': response_text, 'chat_history': chat_history})


if __name__ == "__main__":
    app.run(debug=True)