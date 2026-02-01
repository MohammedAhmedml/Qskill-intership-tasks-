
# Student Marks Analyzer

marks = []  # list to store marks

# Input marks for 5 subjects
for i in range(1, 6):
    mark = int(input(f"Enter marks for subject {i}: "))
    marks.append(mark)

# Calculate total and average
total = sum(marks)
average = total / 5

# Display results
print("\n--- Result ---")
print("Marks:", marks)
print("Total Marks:", total)
print("Average Marks:", average)

# Grade calculation
if average >= 75:
    grade = "Distinction"
elif average >= 60:
    grade = "First Class"
elif average >= 40:
    grade = "Pass"
else:
    grade = "Fail"

print("Grade:", grade)
