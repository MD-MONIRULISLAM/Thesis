import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
df = pd.read_csv('data/employee_survey_responses.csv').drop(columns='Timestamp')
df.columns = [
'Age', 'Gender', 'Industry', 'Job_Level', 'Experience', 'Work_Hours',
'Overtime_Frequency', 'Work_Pressure', 'Daily_Meetings', 'Weekend_Work',
'Stress_Level', 'Daily_Sleep', 'Mental_Fatigue', 'WorkLife_Balance',
'Job_Satisfaction', 'Productivity', 'Task_Completion', 'Absenteeism',
'Emotionally_Exhausted', 'Motivation', 'Consider_Break']
df['Daily_Sleep'] = df['Daily_Sleep'].map(
{'Less than 4 hours': 3, '4–5 hours': 4.5, '6–7 hours': 6.5, '8+ hours': 8.5})
df['Work_Hours'] = df['Work_Hours'].map(
{'Less than 6 hours': 5, '6–8 hours': 7, '8–10 hours': 9,
'10–12 hours': 11, 'More than 12 hours': 13})
df['Daily_Meetings'] = df['Daily_Meetings'].astype(str).map(
{'Less than 1': 0.5, '1': 1, '2': 2, '3': 3, 'More than 3': 4})
df['Overtime_Frequency'] = df['Overtime_Frequency'].map(
{'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Always': 5})
df['Task_Completion'] = df['Task_Completion'].map(
{'Below 50%': 25, '50–70%': 60, '70–90%': 80, 'Above 90%': 95})
df['Absenteeism'] = df['Absenteeism'].map(
{'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Frequently': 3})
df['Emotionally_Exhausted'] = df['Emotionally_Exhausted'].map(
{'Yes': 5, 'No': 1})
# Target: "Yes" to Q19 (emotionally exhausted because of work)
y = (df['Emotionally_Exhausted'] >= 4).astype(int)
survey_features = ['Age', 'Work_Pressure', 'Daily_Meetings', 'WorkLife_Balance',
'Productivity', 'Overtime_Frequency', 'Daily_Sleep',
'Job_Satisfaction', 'Mental_Fatigue', 'Stress_Level',
'Motivation', 'Absenteeism']
df['WorkPressure_x_Meetings'] = df['Work_Pressure'] * df['Daily_Meetings']
df['WorkPressure_x_Sleep'] = df['Work_Pressure'] * df['Daily_Sleep']
df['WorkLife_x_Productivity'] = df['WorkLife_Balance'] * df['Productivity']
df['Overtime_x_Pressure'] = df['Overtime_Frequency'] * df['Work_Pressure']
features = survey_features + ['WorkPressure_x_Meetings', 'WorkPressure_x_Sleep',
'WorkLife_x_Productivity', 'Overtime_x_Pressure']
X = df[features]
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
