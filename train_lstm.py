import os
import json
import pickle

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. SETTINGS
# ============================================================

TRAIN_FILE = "data/train.csv"
VALIDATION_FILE = "data/validation.csv"

MODEL_DIR = "models"

MAX_WORDS = 20000
MAX_LENGTH = 150
EMBEDDING_DIM = 128

LSTM_UNITS = 128

EPOCHS = 10
BATCH_SIZE = 64


# ============================================================
# 2. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print("Training shape:", train_df.shape)
print("Validation shape:", validation_df.shape)

print("\nTraining columns:")
print(train_df.columns.tolist())

print("\nValidation columns:")
print(validation_df.columns.tolist())


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["text", "label"]

for column in required_columns:

    if column not in train_df.columns:
        raise ValueError(
            f"Training CSV does not contain '{column}' column."
        )

    if column not in validation_df.columns:
        raise ValueError(
            f"Validation CSV does not contain '{column}' column."
        )


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

print("\nCleaning text data...")

train_df["text"] = train_df["text"].fillna("").astype(str)
validation_df["text"] = validation_df["text"].fillna("").astype(str)

train_df["label"] = train_df["label"].fillna("Unknown").astype(str)
validation_df["label"] = validation_df["label"].fillna("Unknown").astype(str)


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X_train_text = train_df["text"].values
X_validation_text = validation_df["text"].values

y_train_text = train_df["label"].values
y_validation_text = validation_df["label"].values


print("\nTraining text samples:", len(X_train_text))
print("Validation text samples:", len(X_validation_text))


# ============================================================
# 7. LABEL ENCODING
# ============================================================

print("\n" + "=" * 70)
print("LABEL ENCODING")
print("=" * 70)

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train_text)

# Handle validation labels safely
known_labels = set(label_encoder.classes_)

unknown_validation_labels = [
    label
    for label in y_validation_text
    if label not in known_labels
]

if unknown_validation_labels:

    print(
        "\nWARNING: Validation contains labels not present in training."
    )

    print(
        "Unknown labels:",
        set(unknown_validation_labels)
    )

    raise ValueError(
        "Validation dataset contains labels that are not present "
        "in the training dataset."
    )

y_validation = label_encoder.transform(
    y_validation_text
)

num_classes = len(label_encoder.classes_)

print("Number of classes:", num_classes)

print("\nClasses:")

for index, label in enumerate(label_encoder.classes_):

    print(f"{index} -> {label}")


# ============================================================
# 8. TOKENIZATION
# ============================================================

print("\n" + "=" * 70)
print("TEXT TOKENIZATION")
print("=" * 70)

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train_text)

X_train_sequences = tokenizer.texts_to_sequences(
    X_train_text
)

X_validation_sequences = tokenizer.texts_to_sequences(
    X_validation_text
)

print("Vocabulary size:", len(tokenizer.word_index))


# ============================================================
# 9. PADDING / TRUNCATION
# ============================================================

print("\nPadding sequences...")

X_train = pad_sequences(
    X_train_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_validation = pad_sequences(
    X_validation_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

print("Training input shape:", X_train.shape)
print("Validation input shape:", X_validation.shape)


# ============================================================
# 10. BUILD LSTM MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING LSTM MODEL")
print("=" * 70)

model = Sequential([

    Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM
    ),

    LSTM(
        LSTM_UNITS,
        return_sequences=False
    ),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        num_classes,
        activation="softmax"
    )
])


# ============================================================
# 11. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 12. DISPLAY MODEL
# ============================================================

print("\nModel architecture:")

model.summary()


# ============================================================
# 13. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)


# ============================================================
# 14. TRAIN LSTM
# ============================================================

print("\n" + "=" * 70)
print("STARTING LSTM TRAINING")
print("=" * 70)

history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_validation,
        y_validation
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# 15. SAVE LSTM MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "ticket_classifier_lstm.keras"
)

model.save(model_path)

print("\nLSTM model saved:")
print(model_path)


# ============================================================
# 16. SAVE TOKENIZER
# ============================================================

tokenizer_path = os.path.join(
    MODEL_DIR,
    "lstm_tokenizer.pkl"
)

with open(
    tokenizer_path,
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )

print("Tokenizer saved:")
print(tokenizer_path)


# ============================================================
# 17. SAVE LABEL ENCODER
# ============================================================

encoder_path = os.path.join(
    MODEL_DIR,
    "lstm_label_encoder.pkl"
)

with open(
    encoder_path,
    "wb"
) as file:

    pickle.dump(
        label_encoder,
        file
    )

print("Label encoder saved:")
print(encoder_path)


# ============================================================
# 18. SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    MODEL_DIR,
    "lstm_training_history.json"
)

history_data = {
    key: [
        float(value)
        for value in values
    ]
    for key, values in history.history.items()
}

with open(
    history_path,
    "w"
) as file:

    json.dump(
        history_data,
        file,
        indent=4
    )

print("Training history saved:")
print(history_path)


# ============================================================
# 19. VALIDATION PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING VALIDATION PREDICTIONS")
print("=" * 70)

prediction_probabilities = model.predict(
    X_validation,
    verbose=0
)

predicted_classes = np.argmax(
    prediction_probabilities,
    axis=1
)


# ============================================================
# 20. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_validation,
    predicted_classes,
    target_names=label_encoder.classes_,
    zero_division=0
)

print(report)


# ============================================================
# 21. SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(
    MODEL_DIR,
    "lstm_classification_report.txt"
)

with open(
    report_path,
    "w"
) as file:

    file.write(report)

print("\nClassification report saved:")
print(report_path)


# ============================================================
# 22. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("LSTM TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")

print("1. ", model_path)
print("2. ", tokenizer_path)
print("3. ", encoder_path)
print("4. ", history_path)
print("5. ", report_path)

print("\nYour LSTM model is ready for evaluation.")
