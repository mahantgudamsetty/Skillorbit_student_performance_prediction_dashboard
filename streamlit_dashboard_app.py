import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# import the model we trained in model_training.py
with open ('model/regressor.pkl', 'rb') as regressor_file:
    regressor = pickle.load(regressor_file)

st.set_page_config(
    page_title = 'Student Performance Prediction',
    page_icon = '📊',
    layout = 'wide'
)

st.title('🎓 Student Performance Prediction Dashboard')

# representing boolean values as 0 and 1 for the model to understand

Age=st.slider(
    'Age',
    15, 18, 16)
Gender_string = st.radio('Gender',['Male','Female'])
Gender = 0 if Gender_string=='Male' else 1
Ethnicity_string = st.radio('Ethnicity', ['Caucasian', ' African American', 'Asian', 'Other'])
Ethnicity = 0 if Ethnicity_string == 'Caucasian' else 1 if Ethnicity_string == 'African American' else 2 if Ethnicity_string == 'Asian' else 3 

ParentalEducation_string = st.radio('Parental Education', ['None', 'High School', 'Some College', 'Bachelor Degree', 'Higher'])

ParentalEducation = 0 if ParentalEducation_string == 'None' else 1 if ParentalEducation_string == 'High School' else 2 if ParentalEducation_string == 'Some College' else 3 if ParentalEducation_string == 'Bachelor Degree' else 4
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

Tutoring_string = st.radio('Tutoring', ['Yes', 'No'])
Tutoring = 1 if Tutoring_string == 'Yes' else 0

ParentalSupport_string = st.radio('Parental Support', ['None', 'Low', 'Moderate', 'High', 'Very High'])
ParentalSupport = 0 if ParentalSupport_string == 'None' else 1 if ParentalSupport_string == 'Low' else 2 if ParentalSupport_string == 'Moderate' else 3 if ParentalSupport_string == 'High' else 4

Extracurricular_string = st.radio('Extracurricular', ['Yes', 'No'])
Extracurricular = 1 if Extracurricular_string == 'Yes' else 0

Sports_string = st.radio('Sports', ['Yes', 'No'])
Sports = 1 if Sports_string == 'Yes' else 0

Music_string = st.radio('Music', ['Yes', 'No'])
Music = 1 if Music_string == 'Yes' else 0   

Volunteering_string = st.radio('Volunteering', ['Yes', 'No'])
Volunteering = 1 if Volunteering_string == 'Yes' else 0

# we will make a feature vector with the inputs taken above and then we will use the model to predict the output
# it is going to be a dataframe with one row and 12 columns, each column representing a feature.
feature_vector = pd.DataFrame([[Age, Gender, Ethnicity, ParentalEducation, StudyTimeWeekly, Absences, Tutoring, ParentalSupport, Extracurricular, Sports, Music, Volunteering]],
                              columns=['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering'])


GPA_prediction=regressor.predict(feature_vector)
grade_class='A' if GPA_prediction[0]>=3.5 else 'B' if GPA_prediction[0]>=3.0 else 'C' if GPA_prediction[0]>=2.5 else 'D' if GPA_prediction[0]>=2.0 else 'F'
# a prediction button to trigger the prediction
if st.button('Predict'):
    st.metric('Predicted GPA', round(GPA_prediction[0], 2))
    if grade_class == 'A':
        st.success('🏆 Grade A')
    elif grade_class == 'B':
        st.success('🥈 Grade B')
    elif grade_class == 'C':
        st.info('📘 Grade C')
    elif grade_class == 'D':
        st.warning('⚠️ Grade D')
    else:
        st.error('❌ Grade F')

st.header('📌 Recommendations for Improving Student Performance')

st.write('Based on the students profile, here are some recommendations to improve their performance:')

if(StudyTimeWeekly < 20):
    st.info('- Increase study time to at least 10 hours per week for better results.')
if(Absences > 0):
    st.warning('- Reduce absences to improve academic performance.')
if(Tutoring == 0):
    st.info('- Consider seeking tutoring support to enhance understanding of subjects.')
if(ParentalSupport < 2):
    st.info('- Encourage parental support and involvement in the students education.')
if(Extracurricular == 0):
    st.info('- Engage in extracurricular activities to develop skills and interests.')
if(Sports == 0):
    st.info('- Participate in sports to improve physical health and teamwork skills.')
if(Music == 0):
    st.info('- Explore music or arts programs to enhance creativity and cognitive skills.')    
if(Volunteering == 0):
    st.info('- Get involved in volunteering activities to build social responsibility and empathy.')

feature_names = ['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
importance = pd.Series(regressor.feature_importances_, index=feature_names).sort_values(ascending=False)

st.subheader('📊 Feature Importance')
st.write('Showing feature importances using matplotlib python library')
figu = plt.figure(figsize=(10, 3))
importance.plot(kind='bar', color='skyblue')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance Score')
st.pyplot(figu)