import streamlit as st
import pickle
import pandas as pd
import numpy as np

# import the model we trained in model_training.py
with open ('model/classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

st.title('Student Performance Prediction Dashboard')

# representing boolean values as 0 and 1 for the model to understand

Age=st.slider(
    'Age',
    16, 15, 18)
Gender=st.selectbox(
    'Gender', 
    [0, 1])
Ethnicity = st.selectbox(
    'Ethnicity',
    [0,1,2,3]
)

ParentalEducation = st.selectbox(
    'Parental Education',
    [0,1,2,3,4]
)

StudyTimeWeekly = st.slider(
    'Study Time Weekly',
    0.0,
    20.0,
    5.0
)

Absences = st.slider(
    'Absences',
    0,
    30,
    5
)

Tutoring = st.selectbox(
    'Tutoring',
    [0,1]                  
)

ParentalSupport = st.selectbox(
    'Parental Support',
    [0,1,2,3,4]
)

Extracurricular = st.selectbox(
    'Extracurricular',
    [0,1]
)

Sports = st.selectbox(
    'Sports',
    [0,1]
)

Music = st.selectbox(
    'Music',
    [0,1]
)

Volunteering = st.selectbox(
    'Volunteering',
    [0,1]
)

# we will make a feature vector with the inputs taken above and then we will use the model to predict the output
# it is going to be a dataframe with one row and 12 columns, each column representing a feature.
feature_vector = pd.DataFrame([[Age, Gender, Ethnicity, ParentalEducation, StudyTimeWeekly, Absences, Tutoring, ParentalSupport, Extracurricular, Sports, Music, Volunteering]],
                              columns=['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering'])

prediction=classifier.predict(feature_vector)
st.success(f'The predicted Grade Class for the student is: {prediction[0]}')
