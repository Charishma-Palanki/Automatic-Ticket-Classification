import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "data/train.csv"
VALIDATION_FILE = "data/validation.csv"
TEST_FILE = "data/test.csv"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )

    # Replace non-alphanumeric characters with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AUTOMATIC TICKET CLASSIFICATION - IMPROVED SVM")
print("=" * 70)

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train      : {train_df.shape}")
print(f"Validation : {val_df.shape}")
print(f"Test       : {test_df.shape}")


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = ["text", "label"]

for df_name, df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df)
]:

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"{column} column missing from {df_name}.csv"
            )


# ============================================================
# CLEAN TEXT
# ============================================================

print("\nCleaning text...")

train_text = train_df["text"].fillna("").apply(clean_text)
val_text = val_df["text"].fillna("").apply(clean_text)
test_text = test_df["text"].fillna("").apply(clean_text)

train_labels = train_df["label"].astype(str)
val_labels = val_df["label"].astype(str)
test_labels = test_df["label"].astype(str)


# ============================================================
# LABEL ENCODING
# ============================================================

print("\nEncoding labels...")

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(train_labels)
y_val = label_encoder.transform(val_labels)
y_test = label_encoder.transform(test_labels)

print(
    f"Number of classes: {len(label_encoder.classes_)}"
)


# ============================================================
# WORD TF-IDF
# ============================================================

print("\nCreating WORD TF-IDF features...")

word_vectorizer = TfidfVectorizer(

    lowercase=True,

    stop_words="english",

    ngram_range=(1, 3),

    min_df=2,

    max_df=0.98,

    sublinear_tf=True,

    max_features=180000,

    strip_accents="unicode"
)


X_train_word = word_vectorizer.fit_transform(train_text)

X_val_word = word_vectorizer.transform(val_text)

X_test_word = word_vectorizer.transform(test_text)

print(
    "Word features:",
    X_train_word.shape
)


# ============================================================
# CHARACTER TF-IDF
# ============================================================

print("\nCreating CHARACTER TF-IDF features...")

char_vectorizer = TfidfVectorizer(

    analyzer="char_wb",

    ngram_range=(3, 6),

    min_df=2,

    max_df=0.98,

    sublinear_tf=True,

    max_features=140000
)


X_train_char = char_vectorizer.fit_transform(train_text)

X_val_char = char_vectorizer.transform(val_text)

X_test_char = char_vectorizer.transform(test_text)

print(
    "Character features:",
    X_train_char.shape
)


# ============================================================
# COMBINE FEATURES
# ============================================================

print("\nCombining word and character features...")

X_train = hstack(
    [
        X_train_word,
        X_train_char
    ]
).tocsr()

X_val = hstack(
    [
        X_val_word,
        X_val_char
    ]
).tocsr()

X_test = hstack(
    [
        X_test_word,
        X_test_char
    ]
).tocsr()

print(
    "Combined train features:",
    X_train.shape
)


# ============================================================
# EXPERIMENT WITH SVM SETTINGS
# ============================================================

print("\n")
print("=" * 70)
print("SVM HYPERPARAMETER EXPERIMENTS")
print("=" * 70)

experiments = [

    {
        "C": 1.0,
        "class_weight": None
    },

    {
        "C": 1.5,
        "class_weight": None
    },

    {
        "C": 2.0,
        "class_weight": None
    },

    {
        "C": 2.5,
        "class_weight": None
    },

    {
        "C": 3.0,
        "class_weight": None
    },

    {
        "C": 1.5,
        "class_weight": "balanced"
    },

    {
        "C": 2.0,
        "class_weight": "balanced"
    }
]


results = []

best_model = None
best_c = None
best_weight = None
best_val_accuracy = -1


for i, params in enumerate(experiments, start=1):

    print("\n")
    print("-" * 70)

    print(
        f"Experiment {i}/{len(experiments)}"
    )

    print(
        f"C = {params['C']} | "
        f"class_weight = {params['class_weight']}"
    )

    model = LinearSVC(
        C=params["C"],
        class_weight=params["class_weight"],
        max_iter=5000,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    val_predictions = model.predict(
        X_val
    )

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    val_f1 = f1_score(
        y_val,
        val_predictions,
        average="weighted"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
    )

    print(
        f"Validation Weighted F1: {val_f1:.4f}"
    )

    results.append(
        {
            "C": params["C"],
            "class_weight": params["class_weight"],
            "validation_accuracy": val_accuracy,
            "validation_weighted_f1": val_f1
        }
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model = model

        best_c = params["C"]

        best_weight = params["class_weight"]


# ============================================================
# BEST MODEL
# ============================================================

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Best C: {best_c}"
)

print(
    f"Best class weight: {best_weight}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)


# ============================================================
# TEST SET EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_predictions = best_model.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_weighted_f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted"
)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Weighted F1: "
    f"{test_weighted_f1:.4f}"
)

print(
    f"\nTest Accuracy Percentage: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

test_report = classification_report(
    y_test,
    test_predictions,
    target_names=label_encoder.classes_,
    zero_division=0
)

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(test_report)


# ============================================================
# SAVE MODELS
# ============================================================

print("\n")
print("=" * 70)
print("SAVING MODELS")
print("=" * 70)


joblib.dump(
    best_model,
    os.path.join(
        MODEL_DIR,
        "ticket_classifier_svm.pkl"
    )
)

joblib.dump(
    word_vectorizer,
    os.path.join(
        MODEL_DIR,
        "word_tfidf_vectorizer.pkl"
    )
)

joblib.dump(
    char_vectorizer,
    os.path.join(
        MODEL_DIR,
        "char_tfidf_vectorizer.pkl"
    )
)

joblib.dump(
    label_encoder,
    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl"
    )
)


# Compatibility vectorizer
joblib.dump(
    word_vectorizer,
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.pkl"
    )
)


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    os.path.join(
        MODEL_DIR,
        "classification_report.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AUTOMATIC TICKET CLASSIFICATION\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Best C: {best_c}\n"
    )

    file.write(
        f"Class Weight: {best_weight}\n"
    )

    file.write(
        f"Validation Accuracy: "
        f"{best_val_accuracy:.4f}\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}\n"
    )

    file.write(
        f"Test Weighted F1: "
        f"{test_weighted_f1:.4f}\n\n"
    )

    file.write(
        test_report
    )


# ============================================================
# SAVE EXPERIMENT RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    os.path.join(
        MODEL_DIR,
        "svm_experiment_results.csv"
    ),
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nFinal Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Final Weighted F1: "
    f"{test_weighted_f1:.4f}"
)

print(
    "\nSaved files:"
)

print(
    "models/ticket_classifier_svm.pkl"
)

print(
    "models/word_tfidf_vectorizer.pkl"
)

print(
    "models/char_tfidf_vectorizer.pkl"
)

print(
    "models/label_encoder.pkl"
)

print(
    "models/tfidf_vectorizer.pkl"
)

print(
    "models/classification_report.txt"
)

print(
    "models/svm_experiment_results.csv"
)

print("\nDone!")

