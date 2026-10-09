# ============================================================
# AUTOMATIC TICKET CLASSIFICATION
# PHASE 3 - WORD + CHARACTER TF-IDF + LINEAR SVM
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

print("\n" + "=" * 70)
print("AUTOMATIC TICKET CLASSIFICATION - PHASE 3")
print("WORD + CHARACTER TF-IDF + LINEAR SVM")
print("=" * 70)

# ============================================================
# 2. LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

X_train = train_df["text"].fillna("").astype(str)
y_train = train_df["label"]

X_val = val_df["text"].fillna("").astype(str)
y_val = val_df["label"]

X_test = test_df["text"].fillna("").astype(str)
y_test = test_df["label"]

print("Training samples   :", len(X_train))
print("Validation samples :", len(X_val))
print("Testing samples    :", len(X_test))

# ============================================================
# 3. LOAD LABEL ENCODER
# ============================================================

print("\nLoading label encoder...")

import joblib

label_encoder = joblib.load("models/label_encoder.pkl")

print("Number of classes:", len(label_encoder.classes_))

# ============================================================
# 4. WORD-LEVEL TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: Creating WORD-level TF-IDF features")
print("=" * 70)
word_vectorizer = TfidfVectorizer(
    max_features=120000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode"
)
print("\nFitting word TF-IDF vectorizer...")

X_train_word = word_vectorizer.fit_transform(X_train)

print("Transforming validation data...")
X_val_word = word_vectorizer.transform(X_val)

print("Transforming test data...")
X_test_word = word_vectorizer.transform(X_test)

print("\nWord TF-IDF shapes:")
print("Training   :", X_train_word.shape)
print("Validation :", X_val_word.shape)
print("Testing    :", X_test_word.shape)

# ============================================================
# 5. CHARACTER-LEVEL TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: Creating CHARACTER-level TF-IDF features")
print("=" * 70)

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3,
    max_features=80000,
    sublinear_tf=True
)

print("\nFitting character TF-IDF vectorizer...")

X_train_char = char_vectorizer.fit_transform(X_train)

print("Transforming validation data...")
X_val_char = char_vectorizer.transform(X_val)

print("Transforming test data...")
X_test_char = char_vectorizer.transform(X_test)

print("\nCharacter TF-IDF shapes:")
print("Training   :", X_train_char.shape)
print("Validation :", X_val_char.shape)
print("Testing    :", X_test_char.shape)

# ============================================================
# 6. COMBINE WORD + CHARACTER FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: Combining WORD + CHARACTER features")
print("=" * 70)

X_train_final = hstack([
    X_train_word,
    X_train_char
])

X_val_final = hstack([
    X_val_word,
    X_val_char
])

X_test_final = hstack([
    X_test_word,
    X_test_char
])

print("\nCombined feature shapes:")
print("Training   :", X_train_final.shape)
print("Validation :", X_val_final.shape)
print("Testing    :", X_test_final.shape)

# ============================================================
# 7. SAVE VECTORIZERS
# ============================================================

print("\nSaving Phase 3 vectorizers...")

joblib.dump(
    word_vectorizer,
    "models/word_tfidf_vectorizer_phase3.pkl"
)

joblib.dump(
    char_vectorizer,
    "models/char_tfidf_vectorizer_phase3.pkl"
)

print("Saved:")
print("models/word_tfidf_vectorizer_phase3.pkl")
print("models/char_tfidf_vectorizer_phase3.pkl")

# ============================================================
# 8. TRAIN LINEAR SVM
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: Training Linear SVM")
print("=" * 70)

model = LinearSVC(
    C=2.0,
    class_weight="balanced"
)

print("\nTraining model...")
model.fit(X_train_final, y_train)

print("Training completed successfully.")

# ============================================================
# 9. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: VALIDATION EVALUATION")
print("=" * 70)

val_predictions = model.predict(X_val_final)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

print(
    f"\nValidation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

# ============================================================
# 10. TEST
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: TEST EVALUATION")
print("=" * 70)

test_predictions = model.predict(X_test_final)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    test_predictions,
    target_names=label_encoder.classes_,
    zero_division=0
)

print("\n")
print(report)

report_path = "models/classification_report_phase3.txt"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

print("\nClassification report saved:")
print(report_path)

# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("STEP 8: CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    test_predictions
)

np.save(
    "models/confusion_matrix_phase3.npy",
    cm
)

print(
    "\nConfusion matrix saved:"
    "\nmodels/confusion_matrix_phase3.npy"
)

# ============================================================
# 13. SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 9: SAVING MODEL")
print("=" * 70)

model_path = "models/ticket_classifier_svm_phase3.pkl"

joblib.dump(
    model,
    model_path
)

print("\nModel saved:")
print(model_path)

# ============================================================
# 14. SAVE EVALUATION RESULTS
# ============================================================

results = {
    "phase": "Phase 3",
    "validation_accuracy": float(val_accuracy),
    "test_accuracy": float(test_accuracy),
    "number_of_classes": int(len(label_encoder.classes_)),
    "word_features": int(X_train_word.shape[1]),
    "character_features": int(X_train_char.shape[1]),
    "total_features": int(X_train_final.shape[1]),
    "word_ngram_range": "(1, 2)",
    "character_ngram_range": "(3, 5)",
    "svm_C": 2.0,
    "class_weight": "balanced"
}

results_path = "models/phase3_evaluation_results.pkl"

joblib.dump(
    results,
    results_path
)

print("\nEvaluation results saved:")
print(results_path)

# ============================================================
# 15. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3 TRAINING COMPLETED")
print("=" * 70)

print("\nAccuracy Comparison:")
print("----------------------------------------")
print("Phase 1 Test Accuracy : 68.14%")
print("Phase 2 Test Accuracy : 72.28%")
print(
    f"Phase 3 Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"\nPhase 3 improvement over Phase 2: "
    f"{(test_accuracy - 0.7228) * 100:+.2f}%"
)

print("\nGenerated files:")
print("----------------------------------------")
print("models/word_tfidf_vectorizer_phase3.pkl")
print("models/char_tfidf_vectorizer_phase3.pkl")
print("models/ticket_classifier_svm_phase3.pkl")
print("models/classification_report_phase3.txt")
print("models/confusion_matrix_phase3.npy")
print("models/phase3_evaluation_results.pkl")

print("\nPhase 3 completed successfully!")
print("=" * 70)
