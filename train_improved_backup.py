# ============================================================
# AUTOMATIC TICKET CLASSIFICATION
# PHASE 2 - IMPROVED TF-IDF + LINEAR SVM
# ============================================================

import os
import pickle
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

# ============================================================
# 2. LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

X_train = train_df["text"].fillna("")
y_train = train_df["label"]

X_val = val_df["text"].fillna("")
y_val = val_df["label"]

X_test = test_df["text"].fillna("")
y_test = test_df["label"]

print("Training samples   :", len(X_train))
print("Validation samples :", len(X_val))
print("Testing samples    :", len(X_test))

# ============================================================
# 3. LOAD LABEL ENCODER
# ============================================================

print("\nLoading label encoder...")

# IMPORTANT:
# label_encoder.pkl was created using joblib
import joblib

label_encoder = joblib.load("models/label_encoder.pkl")

print("Number of classes:", len(label_encoder.classes_))

# ============================================================
# 4. CREATE IMPROVED TF-IDF FEATURES
# ============================================================

print("\nCreating improved TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True
)

print("Fitting TF-IDF vectorizer...")

X_train_tfidf = vectorizer.fit_transform(X_train)

print("Transforming validation data...")

X_val_tfidf = vectorizer.transform(X_val)

print("Transforming test data...")

X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF feature shapes:")
print("Training   :", X_train_tfidf.shape)
print("Validation :", X_val_tfidf.shape)
print("Testing    :", X_test_tfidf.shape)

# ============================================================
# 5. SAVE IMPROVED TF-IDF VECTORIZER
# ============================================================

vectorizer_path = "models/tfidf_vectorizer_improved.pkl"

joblib.dump(vectorizer, vectorizer_path)

print("\nImproved TF-IDF vectorizer saved:")
print(vectorizer_path)

# ============================================================
# 6. TRAIN IMPROVED LINEAR SVM
# ============================================================

print("\nTraining improved Linear SVM...")

model = LinearSVC(
    C=1.5,
    class_weight="balanced",
    max_iter=5000
)

model.fit(X_train_tfidf, y_train)

print("Training completed successfully.")

# ============================================================
# 7. VALIDATION EVALUATION
# ============================================================

print("\nEvaluating on validation dataset...")

val_predictions = model.predict(X_val_tfidf)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

print(
    f"\nValidation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

# ============================================================
# 8. TEST EVALUATION
# ============================================================

print("\nEvaluating on test dataset...")

test_predictions = model.predict(X_test_tfidf)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

report = classification_report(
    y_test,
    test_predictions,
    target_names=label_encoder.classes_,
    zero_division=0
)

print(report)

# ============================================================
# 10. SAVE CLASSIFICATION REPORT
# ============================================================

report_path = "models/classification_report_improved.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print("\nClassification report saved:")
print(report_path)

# ============================================================
# 11. CREATE CONFUSION MATRIX
# ============================================================

print("\nCreating confusion matrix...")

cm = confusion_matrix(
    y_test,
    test_predictions
)

np.save(
    "models/confusion_matrix_improved.npy",
    cm
)

print(
    "Confusion matrix data saved to:"
    " models/confusion_matrix_improved.npy"
)

# ============================================================
# 12. SAVE IMPROVED MODEL
# ============================================================

model_path = "models/ticket_classifier_svm_improved.pkl"

joblib.dump(model, model_path)

print("\nImproved model saved to:")
print(model_path)

# ============================================================
# 13. SAVE EVALUATION RESULTS
# ============================================================

results = {
    "validation_accuracy": float(val_accuracy),
    "test_accuracy": float(test_accuracy),
    "original_test_accuracy": 0.6814,
    "accuracy_improvement": float(test_accuracy - 0.6814),
    "number_of_classes": len(label_encoder.classes_),
    "tfidf_features": int(X_train_tfidf.shape[1]),
    "ngram_range": "(1, 2)",
    "min_df": 2,
    "max_df": 0.95,
    "svm_C": 1.5,
    "class_weight": "balanced"
}

results_path = "models/improved_evaluation_results.pkl"

joblib.dump(results, results_path)

print("\nEvaluation results saved:")
print(results_path)

# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2 TRAINING COMPLETED")
print("=" * 60)

print(
    f"Original Test Accuracy : 68.14%"
)

print(
    f"Improved Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Difference             : "
    f"{(test_accuracy - 0.6814) * 100:+.2f}%"
)

print("\nGenerated files:")

print("models/tfidf_vectorizer_improved.pkl")
print("models/ticket_classifier_svm_improved.pkl")
print("models/classification_report_improved.txt")
print("models/confusion_matrix_improved.npy")
print("models/improved_evaluation_results.pkl")

print("\nPhase 2 completed successfully!")
