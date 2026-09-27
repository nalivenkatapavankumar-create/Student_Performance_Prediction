import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("student_data.csv")
print("First 5 rows:")
print(data.head())

print("\n Dataset Shape:")
print(data.shape)

print("\n Dataset Information:")
print(data.info())

print("\n Statistical Summary:")
print(data.describe())

X=data.drop("Final_Score",axis=1)
y=data["Final_Score"]

print("\nInput Features.")
print(X)

print("\n Target: ")
print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print("\n Training data:")
print(X_train)

print("\n Testing Data:")
print(X_test)

print("\n Training Target:")
print(y_train)

print("\n Testing Target:")
print(y_test)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
print("Model Trained Successfully!")

y_pred=model.predict(X_test)
print("\n Predicted Score:")
print(y_pred)

from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_test,y_pred)
print("\n Mean Absolute Error:")
print(mae)

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print("\n R2 Score:")
print(r2)

print("\nEnter Student Details")
study_hours=float(input("Study Hours:"))
attendance=float(input("Attendance:"))
previous_marks=float(input("Previous Marks:"))
assignment_score=float(input("Assignment Score:"))
sleep_hours=float(input("Sleep Hours:"))

new_student=pd.DataFrame({"Study_Hours":[study_hours],"Attendance":[attendance],"Previous_Marks":[previous_marks],"Assignment_Score":[assignment_score],"Sleep_Hours":[sleep_hours]})

new_prediction=model.predict(new_student)

predicted_score = new_prediction[0]

print("\n------------------------------")
print("       STUDENT RESULT")
print("------------------------------")

print(f"Predicted Final Score: {predicted_score:.2f} / 100")

if predicted_score >= 90:
    print("Performance Level: Excellent")
elif predicted_score >= 75:
    print("Performance Level: Very Good")
elif predicted_score >= 60:
    print("Performance Level: Good")
elif predicted_score >= 50:
    print("Performance Level: Average")
else:
    print("Performance Level: Needs Improvement")

print("------------------------------")

# Visualization

features = [
    "Study Hours",
    "Attendance",
    "Previous Marks",
    "Assignment Score",
    "Sleep Hours"
]

values = [
    study_hours,
    attendance,
    previous_marks,
    assignment_score,
    sleep_hours
]

plt.figure(figsize=(10, 5))

bars = plt.bar(features, values)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        str(value),
        ha="center"
    )

plt.title("Student Performance Details")
plt.xlabel("Student Factors")
plt.ylabel("Values")

plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("student_performance_details.png")
plt.show()

# Actual vs Predicted Scores

y_pred = model.predict(X_test)

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.title("Actual vs Predicted Final Scores")
plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()