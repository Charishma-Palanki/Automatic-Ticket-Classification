# ============================================================
# AUTOMATIC TICKET CLASSIFICATION
# PHASE 4 - TUNED WORD + CHARACTER TF-IDF + LINEAR SVM
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. CREATE DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

print("\n" + "=" * 70)
print("AUTOMATIC TICKET CLASSIFICATION - PHASE 4")
print("=" * 70)

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
# label_encoder.pkl was created using joblib,
# so it must also be loaded using joblib.

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

print(
    "Number of classes:",
    len(label_encoder.classes_)
)

# ============================================================
# 4. WORD TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 - WORD TF-IDF")
print("=" * 70)

word_vectorizer = TfidfVectorizer(
    max_features=150000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode"
)

print("\nFitting word TF-IDF vectorizer...")

X_train_word = word_vectorizer.fit_transform(
    X_train
)

print("Transforming validation data...")

X_val_word = word_vectorizer.transform(
    X_val
)

print("Transforming test data...")

X_test_word = word_vectorizer.transform(
    X_test
)

print("\nWord TF-IDF shapes:")
print("Training   :", X_train_word.shape)
print("Validation :", X_val_word.shape)
print("Testing    :", X_test_word.shape)

# ============================================================
# 5. CHARACTER TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - CHARACTER TF-IDF")
print("=" * 70)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=100000,
    sublinear_tf=True
)

print("\nFitting character TF-IDF vectorizer...")

X_train_char = char_vectorizer.fit_transform(
    X_train
)

print("Transforming validation data...")

X_val_char = char_vectorizer.transform(
    X_val
)

print("Transforming test data...")

X_test_char = char_vectorizer.transform(
    X_test
)

print("\nCharacter TF-IDF shapes:")
print("Training   :", X_train_char.shape)
print("Validation :", X_val_char.shape)
print("Testing    :", X_test_char.shape)

# ============================================================
# 6. COMBINE WORD + CHARACTER FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - COMBINING FEATURES")
print("=" * 70)

X_train_combined = hstack(
    [X_train_word, X_train_char]
).tocsr()

X_val_combined = hstack(
    [X_val_word, X_val_char]
).tocsr()

X_test_combined = hstack(
    [X_test_word, X_test_char]
).tocsr()

print("\nCombined feature shapes:")
print("Training   :", X_train_combined.shape)
print("Validation :", X_val_combined.shape)
print("Testing    :", X_test_combined.shape)

# ============================================================
# 7. SAVE VECTORIZERS
# ============================================================

print("\n" + "=" * 70)
print("SAVING PHASE 4 VECTORIZERS")
print("=" * 70)

word_vectorizer_path = (
    "models/word_tfidf_vectorizer_phase4.pkl"
)

char_vectorizer_path = (
    "models/char_tfidf_vectorizer_phase4.pkl"
)

joblib.dump(
    word_vectorizer,
    word_vectorizer_path
)

joblib.dump(
    char_vectorizer,
    char_vectorizer_path
)

print("\nSaved:")
print(word_vectorizer_path)
print(char_vectorizer_path)

# ============================================================
# 8. TRAIN LINEAR SVM
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - TRAINING LINEAR SVM")
print("=" * 70)

model = LinearSVC(
    C=1.5,
    class_weight="balanced"
)

print("\nTraining model...")

model.fit(
    X_train_combined,
    y_train
)

print("Training completed successfully.")

# ============================================================
# 9. VALIDATION EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)

val_predictions = model.predict(
    X_val_combined
)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

print(
    f"\nValidation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

# ============================================================
# 10. TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TEST EVALUATION")
print("=" * 70)

test_predictions = model.predict(
    X_test_combined
)

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
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    test_predictions,
    target_names=label_encoder.classes_,
    zero_division=0
)

print("\n")
print(report)

report_path = (
    "models/classification_report_phase4.txt"
)

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

print("\nCreating confusion matrix...")

cm = confusion_matrix(
    y_test,
    test_predictions
)

cm_path = (
    "models/confusion_matrix_phase4.npy"
)

np.save(
    cm_path,
    cm
)

print(
    "Confusion matrix saved:"
    f" {cm_path}"
)

# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

model_path = (
    "models/ticket_classifier_svm_phase4.pkl"
)

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
    "phase": "Phase 4",

    "validation_accuracy":
        float(val_accuracy),

    "test_accuracy":
        float(test_accuracy),

    "number_of_classes":
        int(len(label_encoder.classes_)),

    "word_features":
        int(X_train_word.shape[1]),

    "character_features":
        int(X_train_char.shape[1]),

    "total_features":
        int(X_train_combined.shape[1]),

    "word_ngram_range":
        "(1,3)",

    "character_ngram_range":
        "(3,5)",

    "word_max_features":
        150000,

    "character_max_features":
        100000,

    "svm_C":
        1.5,

    "class_weight":
        "balanced"
}

results_path = (
    "models/phase4_evaluation_results.pkl"
)

joblib.dump(
    results,
    results_path
)

print("\nEvaluation results saved:")
print(results_path)

# ============================================================
# 15. ACCURACY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("ACCURACY COMPARISON")
print("=" * 70)

phase1 = 0.6814
phase2 = 0.7228
phase3 = 0.7444

print(
    f"\nPhase 1 Test Accuracy : "
    f"{phase1 * 100:.2f}%"
)

print(
    f"Phase 2 Test Accuracy : "
    f"{phase2 * 100:.2f}%"
)

print(
    f"Phase 3 Test Accuracy : "
    f"{phase3 * 100:.2f}%"
)

print(
    f"Phase 4 Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"\nPhase 4 improvement over Phase 3: "
    f"{(test_accuracy - phase3) * 100:+.2f}%"
)

# ============================================================
# 16. GENERATED FILES
# ============================================================

print("\n" + "=" * 70)
print("GENERATED FILES")
print("=" * 70)

print(
    "models/word_tfidf_vectorizer_phase4.pkl"
)

print(
    "models/char_tfidf_vectorizer_phase4.pkl"
)

print(
    "models/ticket_classifier_svm_phase4.pkl"
)

print(
    "models/classification_report_phase4.txt"
)

print(
    "models/confusion_matrix_phase4.npy"
)

print(
    "models/phase4_evaluation_results.pkl"
)

# ============================================================
# 17. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("PHASE 4 COMPLETED SUCCESSFULLY!")
print("=" * 70)

print(
    f"\nFinal Phase 4 Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    "\nNext step: Review Phase 4 accuracy "
    "before starting further tuning."
)
