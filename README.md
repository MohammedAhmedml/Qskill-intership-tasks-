Spam Mail Detector (TF-IDF + Naive Bayes)

Objective
Build a text classification model to detect whether an SMS message is spam or ham (not spam).

Dataset
SMS Spam Collection dataset (UCI). The dataset contains 5,572 messages with two columns:
- label: ham/spam
- message: SMS text

Method / Workflow
1. Loaded the dataset (tab-separated: label and message).
2. Split the data into training and testing sets (80/20) using stratified sampling.
3. Converted text into numerical features using TF-IDF vectorization.
4. Trained a Multinomial Naive Bayes classifier on the TF-IDF vectors.
5. Evaluated the model using accuracy, confusion matrix, and classification report.
6. Saved the confusion matrix plot to outputs/confusion_matrix.png.
7. Tested the model on a few custom demo messages.

Results
- Model: TF-IDF + Multinomial Naive Bayes
- Accuracy: 0.9722 (~97.22%)
- Confusion Matrix [ham, spam]:
  [[966, 0],
   [31, 118]]
- The classification report was printed in the terminal.

How to Run
1. Install dependencies: pip install pandas numpy matplotlib scikit-learn
2. Place dataset file in: data/SMSSpamCollection
3. Run the script: python spam.py

Output Files
- outputs/confusion_matrix.png