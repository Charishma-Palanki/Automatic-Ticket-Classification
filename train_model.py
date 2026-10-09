import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_DIR = "models"
SCREENSHOT_DIR = "screenshots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 75)
print("AUTOMATIC TICKET CLASSIFICATION")
print("WORD + CHARACTER TF-IDF + LINEAR SVM")
print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

print(f"Training samples   : {len(train_df)}")
print(f"Validation samples : {len(val_df)}")
print(f"Testing samples    : {len(test_df)}")


# ============================================================
# PREPARE DATA
# ============================================================

print("\nPreparing text data...")

X_train = (
    train_df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

X_val = (
    val_df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

X_test = (
    test_df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

y_train = train_df["label"].astype(int)
y_val = val_df["label"].astype(int)
y_test = test_df["label"].astype(int)


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

print("\nLoading label encoder...")

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

class_names = label_encoder.classes_

print(
    f"Number of classes: {len(class_names)}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 75)

class_distribution = (
    train_df["label"]
    .value_counts()
    .sort_index()
)

for label, count in class_distribution.items():

    class_name = class_names[int(label)]

    print(
        f"{int(label):2d}  "
        f"{class_name:<50} "
        f"{count}"
    )


# ============================================================
# WORD TF-IDF
# ============================================================

print("\n" + "=" * 75)
print("CREATING WORD-LEVEL TF-IDF")
print("=" * 75)

word_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    sublinear_tf=True,
    max_features=120000
)

X_train_word = word_vectorizer.fit_transform(
    X_train
)

X_val_word = word_vectorizer.transform(
    X_val
)

X_test_word = word_vectorizer.transform(
    X_test
)

print(
    "\nWord TF-IDF shape:"
)

print(
    "Training   :",
    X_train_word.shape
)

print(
    "Validation :",
    X_val_word.shape
)

print(
    "Testing    :",
    X_test_word.shape
)


# ============================================================
# CHARACTER TF-IDF
# ============================================================

print("\n" + "=" * 75)
print("CREATING CHARACTER-LEVEL TF-IDF")
print("=" * 75)

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_df=0.98,
    sublinear_tf=True,
    max_features=100000
)

X_train_char = char_vectorizer.fit_transform(
    X_train
)

X_val_char = char_vectorizer.transform(
    X_val
)

X_test_char = char_vectorizer.transform(
    X_test
)

print(
    "\nCharacter TF-IDF shape:"
)

print(
    "Training   :",
    X_train_char.shape
)

print(
    "Validation :",
    X_val_char.shape
)

print(
    "Testing    :",
    X_test_char.shape
)


# ============================================================
# COMBINE FEATURES
# ============================================================

print("\n" + "=" * 75)
print("COMBINING WORD + CHARACTER FEATURES")
print("=" * 75)

X_train_features = hstack(
    [
        X_train_word,
        X_train_char
    ]
).tocsr()

X_val_features = hstack(
    [
        X_val_word,
        X_val_char
    ]
).tocsr()

X_test_features = hstack(
    [
        X_test_word,
        X_test_char
    ]
).tocsr()

print(
    "\nCombined feature shapes:"
)

print(
    "Training   :",
    X_train_features.shape
)

print(
    "Validation :",
    X_val_features.shape
)

print(
    "Testing    :",
    X_test_features.shape
)


# ============================================================
# SAVE VECTORIZERS
# ============================================================

print("\nSaving vectorizers...")

joblib.dump(
    word_vectorizer,
    "models/word_tfidf_vectorizer.pkl"
)

joblib.dump(
    char_vectorizer,
    "models/char_tfidf_vectorizer.pkl"
)

# Also save the word vectorizer using the old filename
# so the existing project still has its original file.
joblib.dump(
    word_vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print(
    "Saved word_tfidf_vectorizer.pkl"
)

print(
    "Saved char_tfidf_vectorizer.pkl"
)

print(
    "Saved tfidf_vectorizer.pkl"
)


# ============================================================
# LINEAR SVM EXPERIMENTS
# ============================================================

print("\n" + "=" * 75)
print("LINEAR SVM MODEL SELECTION")
print("=" * 75)

experiments = [
    {
        "C": 0.5,
        "class_weight": None
    },
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
        "C": 3.0,
        "class_weight": None
    },
    {
        "C": 1.0,
        "class_weight": "balanced"
    },
    {
        "C": 2.0,
        "class_weight": "balanced"
    }
]


best_model = None
best_config = None
best_val_accuracy = 0.0
best_val_f1 = 0.0

results = []


# ============================================================
# TRAIN MODELS
# ============================================================

for index, config in enumerate(experiments, start=1):

    c_value = config["C"]
    class_weight = config["class_weight"]

    print("\n" + "-" * 75)

    print(
        f"Experiment {index}/{len(experiments)}"
    )

    print(
        f"C = {c_value}"
    )

    print(
        f"class_weight = {class_weight}"
    )

    model = LinearSVC(
        C=c_value,
        class_weight=class_weight,
        max_iter=20000
    )

    print(
        "Training model..."
    )

    model.fit(
        X_train_features,
        y_train
    )

    print(
        "Predicting validation data..."
    )

    val_predictions = model.predict(
        X_val_features
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
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Weighted F1: "
        f"{val_f1:.4f}"
    )

    results.append(
        {
            "C": c_value,
            "class_weight": str(class_weight),
            "validation_accuracy": val_accuracy,
            "validation_weighted_f1": val_f1
        }
    )

    # Primary selection criterion = accuracy
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_val_f1 = val_f1

        best_model = model

        best_config = config


# ============================================================
# DISPLAY EXPERIMENT RESULTS
# ============================================================

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

for result in results:

    print(
        f"C={result['C']:<4} "
        f"Weight={result['class_weight']:<8} "
        f"Accuracy="
        f"{result['validation_accuracy'] * 100:.2f}% "
        f"F1="
        f"{result['validation_weighted_f1']:.4f}"
    )


# ============================================================
# BEST MODEL
# ============================================================

print("\n" + "=" * 75)
print("BEST MODEL SELECTED")
print("=" * 75)

print(
    f"Best C: "
    f"{best_config['C']}"
)

print(
    f"Best class weight: "
    f"{best_config['class_weight']}"
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best Validation Weighted F1: "
    f"{best_val_f1:.4f}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL VALIDATION")
print("=" * 75)

val_predictions = best_model.predict(
    X_val_features
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
    f"\nValidation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Validation Weighted F1: "
    f"{val_f1:.4f}"
)


# ============================================================
# FINAL TESTING
# ============================================================

print("\n" + "=" * 75)
print("FINAL TESTING")
print("=" * 75)

test_predictions = best_model.predict(
    X_test_features
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted"
)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Weighted F1: "
    f"{test_f1:.4f}"
)


# ============================================================
# MOST COMMON MISCLASSIFICATIONS
# ============================================================

print("\n" + "=" * 75)
print("MOST COMMON MISCLASSIFICATIONS")
print("=" * 75)

errors = []

for actual, predicted, text in zip(
    y_test,
    test_predictions,
    X_test
):

    if actual != predicted:

        errors.append(
            (
                class_names[int(actual)],
                class_names[int(predicted)],
                text
            )
        )


error_counts = Counter(
    (
        actual,
        predicted
    )
    for actual, predicted, text in errors
)


print(
    f"\nTotal incorrect predictions: "
    f"{len(errors)}"
)

print(
    "\nTop 20 confusion pairs:"
)

for (
    actual,
    predicted
), count in error_counts.most_common(20):

    print(
        f"\n{actual}"
    )

    print(
        f"   --> {predicted}"
    )

    print(
        f"   Errors: {count}"
    )


# ============================================================
# SAVE MISCLASSIFICATION REPORT
# ============================================================

misclassification_file = (
    "models/misclassification_report.txt"
)

with open(
    misclassification_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "MOST COMMON MISCLASSIFICATIONS\n"
    )

    file.write(
        "=" * 75 + "\n\n"
    )

    file.write(
        f"Total incorrect predictions: "
        f"{len(errors)}\n\n"
    )

    for (
        actual,
        predicted
    ), count in error_counts.most_common(20):

        file.write(
            f"{actual} --> {predicted} "
            f": {count} errors\n"
        )


print(
    "\nSaved:"
)

print(
    misclassification_file
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

report = classification_report(
    y_test,
    test_predictions,
    target_names=class_names,
    zero_division=0
)

print(report)

with open(
    "models/classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nCreating confusion matrix...")

cm = confusion_matrix(
    y_test,
    test_predictions
)

plt.figure(
    figsize=(18, 15)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Ticket Classification Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "screenshots/confusion_matrix.png",
    dpi=200
)

plt.close()

print(
    "Saved:"
)

print(
    "screenshots/confusion_matrix.png"
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 75)
print("SAVING BEST MODEL")
print("=" * 75)

joblib.dump(
    best_model,
    "models/ticket_classifier_svm.pkl"
)

print(
    "Saved:"
)

print(
    "models/ticket_classifier_svm.pkl"
)


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

evaluation_results = {

    "validation_accuracy":
        float(val_accuracy),

    "validation_weighted_f1":
        float(val_f1),

    "test_accuracy":
        float(test_accuracy),

    "test_weighted_f1":
        float(test_f1),

    "number_of_classes":
        len(class_names),

    "training_samples":
        len(train_df),

    "validation_samples":
        len(val_df),

    "testing_samples":
        len(test_df),

    "best_C":
        float(best_config["C"]),

    "best_class_weight":
        str(best_config["class_weight"])
}


joblib.dump(
    evaluation_results,
    "models/evaluation_results.pkl"
)

print(
    "Saved:"
)

print(
    "models/evaluation_results.pkl"
)


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "models/svm_experiment_results.csv",
    index=False
)

print(
    "Saved:"
)

print(
    "models/svm_experiment_results.csv"
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 75)

print(
    f"\nBest C Value: "
    f"{best_config['C']}"
)

print(
    f"Best Class Weight: "
    f"{best_config['class_weight']}"
)

print(
    f"Final Validation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Final Validation Weighted F1: "
    f"{val_f1:.4f}"
)

print(
    f"Final Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Final Test Weighted F1: "
    f"{test_f1:.4f}"
)

print("\nGenerated files:")

print(
    "models/word_tfidf_vectorizer.pkl"
)

print(
    "models/char_tfidf_vectorizer.pkl"
)

print(
    "models/tfidf_vectorizer.pkl"
)

print(
    "models/ticket_classifier_svm.pkl"
)

print(
    "models/label_encoder.pkl"
)

print(
    "models/classification_report.txt"
)

print(
    "models/misclassification_report.txt"
)

print(
    "models/evaluation_results.pkl"
)

print(
    "models/svm_experiment_results.csv"
)

print(
    "screenshots/confusion_matrix.png"
)

print("\nNext step:")

print(
    "Run the Streamlit application and test predictions."
)
