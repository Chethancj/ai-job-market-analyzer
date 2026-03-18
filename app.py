
# ======================================
# AI JOB MARKET SKILL ANALYZER DASHBOARD
# ======================================

import streamlit as st
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np


# ======================================
# PAGE TITLE
# ======================================

st.title("AI Job Market Skill Analyzer")
st.caption("Analyze job market trends, skill demand, and salary insights using ML")

# ======================================
# STEP 1: GENERATE JOB DATA
# ======================================

job_titles = [
"Data Scientist",
"Machine Learning Engineer",
"Data Analyst",
"AI Engineer",
"Business Intelligence Analyst"
]

descriptions = [
"Python SQL machine learning AWS",
"Python TensorFlow deep learning NLP",
"SQL Excel Power BI data analysis",
"Python PyTorch machine learning AWS",
"SQL Tableau Excel analytics"
]

salaries = [55000,65000,48000,70000,52000]

data = []

for i in range(200):

    data.append({
        "job_title": random.choice(job_titles),
        "job_description": random.choice(descriptions),
        "salary": random.choice(salaries)
    })

df = pd.DataFrame(data)

col1, col2 = st.columns(2)

col1.metric("Total Jobs", len(df))
col2.metric("Average Salary", int(df["salary"].mean()))


# ======================================
# STEP 2: SKILL LIST
# ======================================

skills = [
"python",
"sql",
"machine learning",
"aws",
"tensorflow",
"pytorch",
"excel",
"power bi",
"tableau",
"deep learning",
"nlp"
]


# ======================================
# STEP 3: SKILL EXTRACTION
# ======================================

def extract_skills(text):

    text = text.lower()

    found = []

    for skill in skills:

        if skill in text:

            found.append(skill)

    return found


df["skills"] = df["job_description"].apply(extract_skills)

# ======================================
# MACHINE LEARNING MODEL
# ======================================

mlb = MultiLabelBinarizer()

skill_matrix = mlb.fit_transform(df["skills"])

skills_encoded = pd.DataFrame(skill_matrix,
                              columns=mlb.classes_)

model_df = pd.concat([skills_encoded, df["salary"]],axis=1)

X = model_df.drop("salary",axis=1)

y = model_df["salary"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestRegressor()

model.fit(X_train,y_train)

from sklearn.metrics import mean_absolute_error

pred = model.predict(X_test)

error = mean_absolute_error(y_test, pred)

st.write(f"Model MAE: £{int(error)}")

# ======================================
# STEP 4: SKILL DEMAND ANALYSIS
# ======================================

all_skills = []

for skill_list in df["skills"]:

    all_skills.extend(skill_list)

skill_counts = Counter(all_skills)


# ======================================
# DASHBOARD SECTION
# ======================================

st.header("Skill Demand Dashboard")

skills_df = pd.DataFrame(skill_counts.items(),
                         columns=["Skill","Count"])

skills_df = skills_df.sort_values("Count",ascending=False)

st.bar_chart(skills_df.set_index("Skill"))


# ======================================
# SHOW JOB DATA
# ======================================

st.header("Sample Job Postings")

st.dataframe(df.head())



# ======================================
# SALARY PREDICTOR
# ======================================

st.header("Salary Predictor")

selected_skills = st.multiselect(
    "Select Your Skills",
    skills
)

if st.button("Predict Salary"):

    skill_input = mlb.transform([selected_skills])

    prediction = model.predict(skill_input)

    st.success(
        f"Estimated Salary: £{int(prediction[0])}"
    )


# ======================================
# AI CAREER CHATBOT
# ======================================

st.header("AI Career Advisor")

user_question = st.text_input("Ask a career question")


def chatbot(q):
    q = q.lower()

    if "skill" in q:
        return f"Top 5 skills are: {skill_counts.most_common(5)}"

    elif "salary" in q:
        avg = int(df['salary'].mean())
        return f"The average salary is around £{avg}"

    elif "learn" in q:
        return "You should focus on Python, SQL, Machine Learning, and AWS."

    elif "job" in q:
        return "Common roles include Data Scientist, ML Engineer, and Data Analyst."

    else:
        return "Try asking about skills, salary, jobs, or what to learn."
