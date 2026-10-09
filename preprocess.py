from datasets import load_dataset
import pandas as pd
import re
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------python
print("Loading dataset...")

dataset = load_dataset("Tobi-Bueck/customer-support-tickets")
df = dataset["train"].to_pandas()

print(f"Original dataset shape: {df.shape}")


# --------------------------------------------------
# 2. Handle missing values
# --------------------------------------------------
df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")


# --------------------------------------------------
# 3. Combine subject and body
# --------------------------------------------------
df["ticket_text"] = (
    df["subject"].astype(str) + " " + df["body"].astype(str)
)


# --------------------------------------------------
# 4. Text cleaning
# --------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()


df["ticket_text"] = df["ticket_text"].apply(clean_text)


# --------------------------------------------------
# 5. Remove empty tickets
# --------------------------------------------------
df = df[df["ticket_text"].str.len() > 0].copy()


# --------------------------------------------------
# 6. Encode queue labels
# --------------------------------------------------
label_encoder = LabelEncoder()

df["queue_encoded"] = label_encoder.fit_transform(df["queue"])


# --------------------------------------------------
# 7. Display information
# --------------------------------------------------
print("\nProcessed dataset shape:", df.shape)

print("\nQueue classes:")
for number, label in enumerate(label_encoder.classes_):
    print(number, "->", label)

print("\nQueue distribution:")
print(df["queue"].value_counts())


# --------------------------------------------------
# 8. Train / validation / test split
# --------------------------------------------------
X = df["ticket_text"]
y = df["queue_encoded"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\nDataset split:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))


# --------------------------------------------------
# 9. Save processed data
# --------------------------------------------------
train_df = pd.DataFrame({
    "text": X_train,
    "label": y_train
})

val_df = pd.DataFrame({
    "text": X_val,
    "label": y_val
})

test_df = pd.DataFrame({
    "text": X_test,
    "label": y_test
})


train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/validation.csv", index=False)
test_df.to_csv("data/test.csv", index=False)


# --------------------------------------------------
# 10. Save label encoder
# --------------------------------------------------
joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)


print("\nFiles created successfully:")
print("data/train.csv")
print("data/validation.csv")
print("data/test.csv")
print("models/label_encoder.pkl")

print("\nPreprocessing completed successfully!")
