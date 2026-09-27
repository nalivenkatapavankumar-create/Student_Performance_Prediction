# Student Performance Prediction using Machine Learning

## 📌 Project Overview

Student Performance Prediction is a Machine Learning project that predicts a student's final score based on different academic and lifestyle factors.

The project uses **Linear Regression** to predict the final score of a student.

## 🎯 Objective

The main objective of this project is to analyze student-related factors and predict their expected final academic score using Machine Learning.

## 📊 Input Features

The model uses the following five features:

- Study Hours
- Attendance
- Previous Marks
- Assignment Score
- Sleep Hours

## 🎯 Target Variable

- Final Score

## 🛠️ Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Linear Regression

## 🔄 Machine Learning Workflow

1. Load the student dataset using Pandas.
2. Explore the dataset using `head()`, `shape`, `info()` and `describe()`.
3. Separate input features and target variable.
4. Split the dataset into training and testing data.
5. Train a Linear Regression model.
6. Predict final scores using the trained model.
7. Evaluate the model using MAE and R² Score.
8. Accept new student details from the user.
9. Predict the final score for the new student.
10. Classify the student's performance level.
11. Visualize the student factors and prediction results.

## 📈 Model Evaluation

The model was evaluated using:

### Mean Absolute Error (MAE)

MAE measures the average difference between the actual scores and predicted scores.

**MAE: Approximately 0.93**

### R² Score

R² Score indicates how well the model explains the variation in the target values.

**R² Score: Approximately 0.995**

## 👨‍🎓 Sample Prediction

For a sample student with:

- Study Hours: 7
- Attendance: 85
- Previous Marks: 80
- Assignment Score: 88
- Sleep Hours: 7

The model predicted:

**Final Score: 86.38 / 100**

**Performance Level: Very Good**

## 📊 Visualizations

### Student Performance Details

The bar chart displays the values of the five input factors for a student.

### Actual vs Predicted Final Scores

The scatter plot compares the actual final scores with the scores predicted by the Linear Regression model.

## 📁