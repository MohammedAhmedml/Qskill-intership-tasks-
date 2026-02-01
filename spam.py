import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    # Path to dataset
    data_path = os.path.join("data", "SMSSpamCollection")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Place the file inside Task2_Spam_Detector/data/ and name it SMSSpamCollection."
        )

    # Load data (tab-separated: label \t message)
    df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "message"])

    print("Dataset shape:", df.shape)
    print("\nLabel counts:\n", df["label"].value_counts())

    # Split
    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Predict
    y_pred = model.predict(X_test_vec)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    report = classification_report(y_test, y_pred)

    print("\nModel: TF-IDF + Multinomial Naive Bayes")
    print("Accuracy:", acc)
    print("\nConfusion Matrix [ham, spam]:\n", cm)
    print("\nClassification Report:\n", report)

    # Save confusion matrix image
    os.makedirs("outputs", exist_ok=True)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Spam Detector)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["ham", "spam"])
    plt.yticks([0, 1], ["ham", "spam"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    out_path = os.path.join("outputs", "confusion_matrix.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {out_path}")

    # Demo predictions
    demo_messages = [
        "Congratulations! You've won a free gift card. Click here to claim now!",
        "Hey, are we still meeting at 6 pm today?",
        "URGENT! Your account has been compromised. Verify your details immediately.",
        "Can you send me the notes from class?",
        "Win cash prizes!!! Reply WIN to enter the contest."
    ]

    demo_pred = model.predict(vectorizer.transform(demo_messages))

    print("\nDemo Predictions:")
    for msg, pred in zip(demo_messages, demo_pred):
        print(f"- {pred.upper():4} | {msg}")


if __name__ == "__main__":
    main()