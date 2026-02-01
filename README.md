Iris Flower Classification (KNN)

Objective
Build a machine learning model to classify Iris flowers into three species (setosa, versicolor, virginica) using sepal and petal measurements.

Dataset
Scikit-learn built-in Iris dataset (sklearn.datasets.load_iris) containing 150 samples, 4 numeric features, and 3 classes.

Method / Workflow
1. Loaded the Iris dataset and created a DataFrame for basic exploration (EDA).
2. Displayed sample rows and checked class distribution.
3. Visualized the dataset using a scatter plot of petal length vs petal width and saved it to outputs/iris_scatter.png.
4. Split the dataset into training and testing sets (80/20) using stratified sampling.
5. Trained a K-Nearest Neighbors classifier with k = 5.
6. Evaluated the model using accuracy, confusion matrix, and classification report.
7. Performed a demo prediction on one test sample.

Results
- Model: K-Nearest Neighbors (k=5)
- Accuracy: 1.00
- Confusion matrix and classification report were printed in the terminal output.

How to Run
1. Install dependencies: pip install pandas numpy matplotlib scikit-learn
2. Run the script: python iris.py