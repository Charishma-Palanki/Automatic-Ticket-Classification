
import re
import joblib
import streamlit as st
from scipy.sparse import hstack
from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Automatic Ticket Classification",
    page_icon="🎫",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎫 Automatic Ticket Classification")

st.markdown(
    "AI-powered customer support ticket classification "
    "using Word TF-IDF, Character TF-IDF, Linear SVM, "
    "and Groq for customer replies."
)


# ============================================================
# MODEL FILE PATHS
# ============================================================

MODEL_PATH = "models/ticket_classifier_svm.pkl"
WORD_VECTORIZER_PATH = "models/word_tfidf_vectorizer.pkl"
CHAR_VECTORIZER_PATH = "models/char_tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    word_vectorizer = joblib.load(WORD_VECTORIZER_PATH)
    char_vectorizer = joblib.load(CHAR_VECTORIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    return (
        model,
        word_vectorizer,
        char_vectorizer,
        label_encoder
    )


try:
    (
        model,
        word_vectorizer,
        char_vectorizer,
        label_encoder
    ) = load_models()

except Exception as e:
    st.error(f"Unable to load trained model files: {e}")
    st.stop()


# ============================================================
# GROQ API CONNECTION
# ============================================================

groq_client = None
groq_connected = False
groq_error = ""

try:
    api_key = st.secrets["GROQ_API_KEY"]
    groq_client = Groq(api_key=api_key)
    groq_connected = True

except Exception as e:
    groq_error = str(e)


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Keep letters, numbers and spaces
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# TICKET PREDICTION
# ============================================================

def predict_ticket(ticket):
    cleaned_ticket = clean_text(ticket)

    word_features = word_vectorizer.transform(
        [cleaned_ticket]
    )

    char_features = char_vectorizer.transform(
        [cleaned_ticket]
    )

    combined_features = hstack(
        [word_features, char_features]
    ).tocsr()

    prediction = model.predict(combined_features)

    predicted_queue = label_encoder.inverse_transform(
        prediction
    )[0]

    return predicted_queue


# ============================================================
# GROQ CUSTOMER REPLY
# ============================================================

def generate_customer_reply(ticket, predicted_queue):

    if not groq_connected or groq_client is None:
        return (
            "Groq API is not connected. "
            "Check your .streamlit/secrets.toml file. "
            f"Details: {groq_error}"
        )

    prompt = f"""
Customer Support Ticket:
{ticket}

Predicted Support Queue:
{predicted_queue}

Write a professional and polite customer-facing response.

Requirements:
- Acknowledge the customer's issue.
- Show empathy.
- Address the concern clearly and concisely.
- Do not mention AI, Groq, or machine learning.
- Do not mention the internal classification system.
- Do not mention the predicted support queue.
- Do not invent refunds or compensation.
- Do not invent company policies or timelines.
- Do not make unsupported promises.
- End politely.

Generate only the customer-facing response.
"""

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional customer support agent. "
                        "Write clear, polite, and helpful responses. "
                        "Never invent company policies or promises."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_completion_tokens=300
        )

        reply = response.choices[0].message.content

        if reply:
            return reply.strip()

        return "Groq did not return any text."

    except Exception as e:
        return f"Groq generation error: {str(e)}"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("📊 Project Information")

    st.write("**Classification Model:** Linear SVM")
    st.write("**Features:** Word TF-IDF + Character TF-IDF")
    st.write("**Classes:** 52")
    st.write("**Test Accuracy:** 78.50%")
    st.write("**Weighted F1 Score:** 0.7846")

    st.divider()

    if groq_connected:
        st.success("✅ Groq API Key Loaded")
    else:
        st.error("❌ Groq API Not Connected")
        st.caption(groq_error)


# ============================================================
# CUSTOMER TICKET INPUT
# ============================================================

st.subheader("📝 Enter Customer Support Ticket")

ticket = st.text_area(
    "Customer Ticket",
    height=180,
    placeholder="Enter the customer's support issue here..."
)


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("🔍 Predict Ticket Queue", use_container_width=True):

    if not ticket.strip():
        st.warning("Please enter a customer support ticket.")

    else:
        try:
            predicted_queue = predict_ticket(ticket)

            st.session_state["ticket"] = ticket
            st.session_state["predicted_queue"] = predicted_queue

            st.success("Ticket classification completed.")

            st.subheader("🎯 Predicted Support Queue")
            st.info(predicted_queue)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Test Accuracy", "78.50%")

            with col2:
                st.metric("Weighted F1", "0.7846")

        except Exception as e:
            st.error(f"Ticket prediction failed: {e}")


# ============================================================
# GROQ CUSTOMER REPLY SECTION
# ============================================================

st.divider()

st.subheader("🤖 Generate Customer Reply using Groq")

st.write(
    "After predicting the support queue, generate "
    "a professional customer-facing response."
)


# ============================================================
# GENERATE REPLY BUTTON
# ============================================================

if "predicted_queue" in st.session_state:

    if st.button(
        "✨ Generate Customer Reply",
        use_container_width=True
    ):
        with st.spinner("Generating customer reply..."):

            reply = generate_customer_reply(
                st.session_state["ticket"],
                st.session_state["predicted_queue"]
            )

        st.subheader("💬 Customer Reply")

        if reply.startswith(
            ("Groq generation error:", "Groq API is not connected.")
        ):
            st.error(reply)

        else:
            st.success("Customer reply generated successfully.")
            st.write(reply)

else:
    st.info(
        "First enter a customer ticket and click "
        "'Predict Ticket Queue'."
    )


# ============================================================
# PROJECT WORKFLOW
# ============================================================

st.divider()

st.subheader("🔄 Project Workflow")

st.markdown(
    """
**Customer Ticket**

↓

**Text Preprocessing**

↓

**Word TF-IDF + Character TF-IDF**

↓

**Linear SVM**

↓

**Predicted Support Queue**

↓

**Groq Language Model**

↓

**Professional Customer Reply**
"""
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Automatic Ticket Classification using TF-IDF + Linear SVM "
    "with Groq-powered customer reply generation."
)

