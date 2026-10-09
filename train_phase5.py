# ============================================================
# AUTOMATIC TICKET CLASSIFICATION
# PHASE 5 - TUNED WORD + CHARACTER TF-IDF + LINEAR SVM
# ============================================================

import os
import pickle
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
# 1. DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

print("\n" + "=" * 75)
print("AUTOMATIC TICKET CLASSIFICATION - PHASE 5")
print("=" * 75)

# ============================================================
# 2. LOAD DATA
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

label_encoder = None

try:
    import joblib

    label_encoder = joblib.load(
        "models/label_encoder.pkl"
    )

except Exception as e:
    print("Joblib loading failed:", e)
    print("Trying pickle loading...")

    try:
        with open(
            "models/label_encoder.pkl",
            "rb"
        ) as f:
            label_encoder = pickle.load(f)

    except Exception as e2:
        print("Pickle loading also failed:", e2)
        raise RuntimeError(
            "Could not load models/label_encoder.pkl"
        )

print(
    "Number of classes:",
    len(label_encoder.classes_)
)

# ============================================================
# 4. WORD TF-IDF
# ============================================================

print("\n" + "=" * 75)
print("STEP 1 - WORD TF-IDF")
print("=" * 75)

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

print("\n" + "=" * 75)
print("STEP 2 - CHARACTER TF-IDF")
print("=" * 75)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_df=0.98,
    max_features=120000,
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
# 6. COMBINE FEATURES
# ============================================================

print("\n" + "=" * 75)
print("STEP 3 - COMBINING WORD + CHARACTER FEATURES")
print("=" * 75)

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

print("\nSaving Phase 5 vectorizers...")

with open(
    "models/word_tfidf_vectorizer_phase5.pkl",
    "wb"
) as f:
    pickle.dump(
        word_vectorizer,
        f
    )

with open(
    "models/char_tfidf_vectorizer_phase5.pkl",
    "wb"
) as f:
    pickle.dump(
        char_vectorizer,
        f
    )

print("Saved:")
print(
    "models/word_tfidf_vectorizer_phase5.pkl"
)
print(
    "models/char_tfidf_vectorizer_phase5.pkl"
)

# ============================================================
# 8. TRAIN TUNED LINEAR SVM
# ============================================================

print("\n" + "=" * 75)
print("STEP 4 - TRAINING TUNED LINEAR SVM")
print("=" * 75)

print("\nSVM parameters:")
print("C              : 2.0")
print("class_weight   : balanced")
print("dual           : auto")
print("max_iter       : 5000")

model = LinearSVC(
    C=2.0,
    class_weight="balanced",
    dual="auto",
    max_iter=5000
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

print("\n" + "=" * 75)
print("VALIDATION EVALUATION")
print("=" * 75)

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

print("\n" + "=" * 75)
print("TEST EVALUATION")
print("=" * 75)

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

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

report = classification_report(
    y_test,
    test_predictions,
    target_names=label_encoder.classes_,
    zero_division=0
)

print("\n")
print(report)

report_path = (
    "models/classification_report_phase5.txt"
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
    "models/confusion_matrix_phase5.npy"
)

np.save(
    cm_path,
    cm
)

print("Confusion matrix saved:")
print(cm_path)

# ============================================================
# 13. SAVE MODEL
# ============================================================

model_path = (
    "models/ticket_classifier_svm_phase5.pkl"
)

with open(
    model_path,
    "wb"
) as f:
    pickle.dump(
        model,
        f
    )

print("\nModel saved:")
print(model_path)

# ============================================================
# 14. SAVE EVALUATION RESULTS
# ============================================================

results = {
    "phase": "Phase 5",
    "validation_accuracy": float(
        val_accuracy
    ),
    "test_accuracy": float(
        test_accuracy
    ),
    "number_of_classes": int(
        len(label_encoder.classes_)
    ),
    "word_features": int(
        X_train_word.shape[1]
    ),
    "character_features": int(
        X_train_char.shape[1]
    ),
    "total_features": int(
        X_train_combined.shape[1]
    ),
    "word_ngram_range": "(1,3)",
    "character_ngram_range": "(3,5)",
    "word_max_features": 150000,
    "character_max_features": 120000,
    "svm_C": 2.0,
    "class_weight": "balanced"
}

results_path = (
    "models/phase5_evaluation_results.pkl"
)

with open(
    results_path,
    "wb"
) as f:
    pickle.dump(
        results,
        f
    )

print("\nEvaluation results saved:")
print(results_path)

# ============================================================
# 15. ACCURACY COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("ACCURACY COMPARISON")
print("=" * 75)

phase1 = 0.6814
phase2 = 0.7228
phase3 = 0.7444
phase4 = 0.7533

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
    f"{phase4 * 100:.2f}%"
)

print(
    f"Phase 5 Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"\nPhase 5 improvement over Phase 4: "
    f"{(test_accuracy - phase4) * 100:+.2f}%"
)

# ============================================================
# 16. GENERATED FILES
# ============================================================

print("\n" + "=" * 75)
print("GENERATED FILES")
print("=" * 75)

print(
    "models/word_tfidf_vectorizer_phase5.pkl"
)

print(
    "models/char_tfidf_vectorizer_phase5.pkl"
)

print(
    "models/ticket_classifier_svm_phase5.pkl"
)

print(
    "models/classification_report_phase5.txt"
)

print(
    "models/confusion_matrix_phase5.npy"
)

print(
    "models/phase5_evaluation_results.pkl"
)

# ============================================================
# 17. FINAL RESULT
# ============================================================

print("\n" + "=" * 75)
print("PHASE 5 COMPLETED SUCCESSFULLY!")
print("=" * 75)

print(
    f"\nFinal Phase 5 Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

if test_accuracy > phase4:
    print(
        "\nPhase 5 improved the model."
    )
else:
    print(
        "\nPhase 5 did not improve over Phase 4."
    )

print("\nNext step:")
print(
    "Use the best-performing phase for "
    "the Streamlit application."
)
