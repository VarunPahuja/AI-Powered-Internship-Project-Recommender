# 🎓 Internship Recommendation System

A Streamlit-based application that matches students with internship opportunities based on their skills, interests, and resume information.

## 📋 Overview

This system helps students find relevant internship opportunities by analyzing their skills and interests against available internship listings. It uses natural language processing and machine learning to provide personalized recommendations.

## ✨ Features

- **Resume Parsing**: Automatically extract skills and interests from uploaded PDF resumes
- **Intelligent Matching**: Uses both TF-IDF similarity and machine learning models for recommendations
- **Interactive UI**: Clean Streamlit interface with detailed internship information
- **Feedback System**: Collects and stores user feedback for future recommendation improvements
- **Data Visualization**: Visual representation of top internship matches

## 🔧 Technologies Used

- **Streamlit**: For the web application interface
- **pandas**: For data manipulation and analysis
- **pdfplumber**: For extracting text from resumes
- **scikit-learn**: For TF-IDF vectorization, cosine similarity, and RandomForest model
- **Altair**: For data visualization

## 📁 Repository Structure

- `internship_app.py`: Main Streamlit application
- `internship_scraper.py`: Script for scraping internship data
- `internships.csv`: Database of internship opportunities
- `cleaned_jobs.csv`: Processed job listings
- `scraped_internships (naukri).csv`: Raw data scraped from job portals

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit pandas pdfplumber scikit-learn altair
```

### Running the Application

1. Clone the repository:
```bash
git clone <repository-url>
cd internship-recommender
```

2. Run the Streamlit app:
```bash
streamlit run internship_app.py
```

3. Open your browser and navigate to `http://localhost:8501`

## 💻 Usage

1. **Upload Resume**: Upload a PDF resume to automatically extract skills and interests
2. **Manual Entry**: Alternatively, manually enter your skills and interests
3. **View Recommendations**: The system will display personalized internship recommendations
4. **Provide Feedback**: Rate recommendations to help improve future matches

## 🧠 How it Works

The recommendation system uses a two-step approach:

1. **Initial Filtering**: Filters internships based on matching skills
2. **Similarity Computation**:
   - TF-IDF vectorization to calculate text similarity between user profile and internship descriptions
   - Machine learning model (RandomForest) trained on features like number of skills, perks, etc.
   - Combined score from both methods for final ranking

## 🔄 Data Pipeline

1. Internship data is scraped from job portals
2. Data is cleaned and standardized
3. User inputs skills or uploads resume
4. System matches and ranks internships
5. Results are presented with visualization

## 📊 Future Improvements

- Implement more advanced NLP techniques for better matching
- Add collaborative filtering based on user feedback
- Expand resume parsing capabilities
- Add more data sources for internship listings


## 👨‍💻 Contributors

- Varun Pahuja
- Ekaansh Sawaria
- Ishan Anand
- Abhinav Mishra

---

Feel free to contribute or report issues!
