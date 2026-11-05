
import numpy as np
import pandas as pd
import os

# Set random seed
np.random.seed(42)

# Number of student records
number_of_students = 1000

# Generate student data
attendance = np.random.uniform(0, 100, number_of_students)
study_hours = np.random.uniform(0, 10, number_of_students)
previous_marks = np.random.uniform(0, 100, number_of_students)
assignment_marks = np.random.uniform(0, 100, number_of_students)
quiz_marks = np.random.uniform(0, 100, number_of_students)

# Calculate performance score
score = (
    attendance * 0.20 +
    study_hours * 2.0 +
    previous_marks * 0.25 +
    assignment_marks * 0.20 +
    quiz_marks * 0.15
)

# Classify student performance
def classify_performance(score):
    if score < 50:
        return 0
    elif score < 70:
        return 1
    else:
        return 2

performance = [
    classify_performance(value) for value in score
]

# Create DataFrame
data = pd.DataFrame({
    "attendance": attendance,
    "study_hours": study_hours,
    "previous_marks": previous_marks,
    "assignment_marks": assignment_marks,
    "quiz_marks": quiz_marks,
    "performance": performance
})

# Save dataset
os.makedirs("dataset", exist_ok=True)

data.to_csv(
    "dataset/student_performance.csv",
    index=False
)

print("Dataset created successfully!")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nPerformance Distribution:")
print(data["performance"].value_counts().sort_index())
