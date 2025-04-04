import streamlit as st
import pandas as pd
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
import altair as alt

# --- Utility Functions ---

@st.cache_data
def load_internship_data(file_path):
    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        st.error("CSV file is empty.")
        return None
    except pd.errors.ParserError:
        st.error("Error parsing CSV file.")
        return None
    except pd.errors.DtypeWarning:
        st.warning("Dtype warning: check your CSV file for mixed types.")
        return None
    except FileNotFoundError:
        st.error("CSV file not found.")
        return None

def preprocess_internships(df):
    df = df.copy()
    skill_cols = [f'Skill {i}' for i in range(1, 7)]
    perk_cols = [f'Perk {i}' for i in range(1, 7)]
    df[skill_cols] = df[skill_cols].fillna('')
    df[perk_cols] = df[perk_cols].fillna('')
    df['skills'] = df[skill_cols].apply(lambda x: ', '.join(x), axis=1)
    df['perks'] = df[perk_cols].apply(lambda x: ', '.join(x), axis=1)
    df['Job Title'] = df['Job Title'].str.lower()
    df['Company'] = df['Company'].str.lower()
    df['Location'] = df['Location'].str.lower()
    return df

def parse_resume(resume_file):
    skills_keywords = [
        'python', 'java', 'sql', 'data analysis', 'machine learning', 'deep learning',
        'tensorflow', 'pytorch', 'excel', 'git', 'docker', 'nlp', 'keras', 'flask'
    ]
    interests_keywords = ['AI', 'software engineering', 'data science', 'web development', 'backend', 'frontend']

    with pdfplumber.open(resume_file) as pdf:
        text = "\n".join([page.extract_text() or '' for page in pdf.pages])

    name = re.findall(r"Name\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
    skills = [kw for kw in skills_keywords if kw.lower() in text.lower()]
    interests = [kw for kw in interests_keywords if kw.lower() in text.lower()]

    return {
        "name": name[0] if name else "John Doe",
        "skills": ", ".join(skills),
        "interests": ", ".join(interests)
    }

def filter_internships(df, student):
    df = df.copy()
    if student.get('skills'):
        student_skills = set(student['skills'].lower().split(', '))
        df['skill_match'] = df['skills'].apply(lambda x: len(set(x.lower().split(', ')) & student_skills))
        df = df[df['skill_match'] > 0]
    return df

def compute_similarity(df, student):
    df = df.copy()
    query_parts = []
    if student['skills']:
        query_parts.append(student['skills'])
    if student['interests']:
        query_parts.append(student['interests'])
    if not query_parts:
        return df.assign(similarity=0)

    student_query = " ".join(query_parts)
    df['text'] = df['Job Title'] + " " + df['Company'] + " " + df['skills']
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['text'].tolist() + [student_query])
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    df['similarity'] = similarity_scores.flatten()
    return df.sort_values(by='similarity', ascending=False)

def train_similarity_model(df, student):
    df = compute_similarity(df, student)
    df['label'] = (df['similarity'] > df['similarity'].mean()).astype(int)
    df['num_skills'] = df['skills'].apply(lambda x: len(x.split(', ')))
    df['num_perks'] = df['perks'].apply(lambda x: len(x.split(', ')))
    df['location_len'] = df['Location'].apply(len)
    df['jobtitle_len'] = df['Job Title'].apply(len)
    features = ['num_skills', 'num_perks', 'location_len', 'jobtitle_len']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[features], df['label'])
    return model

def ml_predict_similarity(df, model):
    df['num_skills'] = df['skills'].apply(lambda x: len(x.split(', ')))
    df['num_perks'] = df['perks'].apply(lambda x: len(x.split(', ')))
    df['location_len'] = df['Location'].apply(len)
    df['jobtitle_len'] = df['Job Title'].apply(len)
    features = ['num_skills', 'num_perks', 'location_len', 'jobtitle_len']
    df['ml_similarity'] = model.predict(df[features])
    return df

def combine_scores(df):
    df['combined_score'] = 0.5 * df['similarity'] + 0.5 * df['ml_similarity']
    return df.sort_values(by='combined_score', ascending=False)

def plot_similarity_chart(df):
    top = df.head(5)[['Job Title', 'combined_score']].copy()
    top['Job Title'] = top['Job Title'].str.title()
    chart = alt.Chart(top).mark_bar().encode(
        x=alt.X('combined_score:Q', title='Combined Similarity Score'),
        y=alt.Y('Job Title:N', sort='-x'),
        tooltip=['Job Title', 'combined_score']
    ).properties(title="Top 5 Internship Matches", height=300)
    return chart

feedback_store = {}

def collect_feedback(student_name, job_title, feedback_type):
    key = f"{student_name}_{job_title}"
    feedback_store[key] = feedback_type

# --- Streamlit Interface ---

st.set_page_config(page_title="Internship Recommender", layout="wide")
st.title("\U0001F4C4 Internship Recommendation System")

st.sidebar.header("Upload Resume or Enter Skills")
resume_file = st.sidebar.file_uploader("Upload PDF Resume (optional)", type=['pdf'])

internships = load_internship_data("internships.csv")

if internships is not None:
    internships = preprocess_internships(internships)

    student = {
        "name": "Student",
        "skills": "",
        "interests": ""
    }

    if resume_file:
        student = parse_resume(resume_file)
        st.sidebar.markdown("🧾 **Parsed from Resume**")

    student['name'] = st.sidebar.text_input("👤 Name", value=student['name'])
    student['skills'] = st.sidebar.text_input("🛠 Skills (comma separated)", value=student['skills'])
    student['interests'] = st.sidebar.text_input("💡 Interests (comma separated)", value=student['interests'])

    if student['skills'] or student['interests']:
        with st.spinner("🔍 Matching internships..."):
            filtered = filter_internships(internships, student)

            if filtered.empty:
                st.warning("❌ No internships match your preferences.")
            else:
                model = train_similarity_model(filtered, student)
                predicted = compute_similarity(filtered, student)
                predicted = ml_predict_similarity(predicted, model)
                combined = combine_scores(predicted)

                st.success("✅ Top Internship Recommendations (Combined Model)")
                st.altair_chart(plot_similarity_chart(combined), use_container_width=True)

                top5 = combined.head(5)[[
                    'Job Title', 'Company', 'Location', 'Job Type', 'Experience', 'skills', 'perks', 'combined_score'
                ]].reset_index(drop=True)

                for i, row in top5.iterrows():
                    with st.expander(f"🔹 {row['Job Title'].title()} at {row['Company'].title()}"):
                        st.write(f"**Location:** {row['Location'].title()}")
                        st.write(f"**Job Type:** {row['Job Type']}")
                        st.write(f"**Experience:** {row['Experience']}")
                        st.write(f"**Skills:** {row['skills']}")
                        st.write(f"**Perks:** {row['perks']}")
                        st.write(f"**Match Score:** {row['combined_score']:.2f}")

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"👍 Interested - {i}", key=f"like_{i}"):
                                collect_feedback(student['name'], row['Job Title'], 'positive')
                                st.success("Thanks for your feedback!")
                        with col2:
                            if st.button(f"👎 Not Interested - {i}", key=f"dislike_{i}"):
                                collect_feedback(student['name'], row['Job Title'], 'negative')
                                st.info("Feedback recorded.")
    else:
        st.info("Please enter at least **skills** or **interests** to get recommendations.")
else:
    st.error("❌ Unable to load internship data.")
