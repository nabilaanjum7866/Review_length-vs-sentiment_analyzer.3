import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Review Length vs Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "review_text" not in st.session_state:
    st.session_state.review_text = ""
if "example_select" not in st.session_state:
    st.session_state.example_select = "Select an example"

# ---------------- RULE BASED ----------------
def rule_based_fix(text):
    text = text.lower()

    negative_patterns = [
        "not good", "not worth", "not nice", "not recommend",
        "not a good", "not a great", "never", "no good",
        "bad", "worst", "waste", "terrible", "awful", "boring"
    ]

    neutral_patterns = [
        "somewhat", "average", "okay", "fine",
        "not bad", "not good enough", "not completely",
        "could be better", "just okay"
    ]

    positive_patterns = [
        "good", "great", "amazing", "excellent",
        "best", "love", "fantastic", "awesome"
    ]

    for pattern in negative_patterns:
        if pattern in text:
            return "negative"

    for pattern in neutral_patterns:
        if pattern in text:
            return "neutral"

    for pattern in positive_patterns:
        if pattern in text:
            return "positive"

    return None

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.result-box {
    padding: 20px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
}
.result-box.positive {background-color: #28a745; color:white;}
.result-box.negative {background-color: #dc3545; color:white;}
.result-box.neutral {background-color: #ffc107; color:black;}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center'>🎬 Review Length vs Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Analyze movie reviews and understand sentiment instantly.</p>", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    df = pd.read_csv("small dataset.csv").head(1000)

    df['review'] = df['review'].apply(lambda x: re.sub(r'[^a-zA-Z ]', ' ', x).lower())
    df['length'] = df['review'].apply(len)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return df, vectorizer, model, acc

df, vectorizer, model, acc = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Menu")

with st.sidebar.expander("⚙️ Settings", expanded=True):
    show_keywords = st.checkbox("Show Keywords", True)
    show_confidence = st.checkbox("Show Confidence", True)

with st.sidebar.expander("📊 Dataset Info"):
    st.write(f"Total Reviews Loaded: {len(df)}")
    st.write(f"Positive Reviews: {len(df[df['sentiment']=='positive'])}")
    st.write(f"Negative Reviews: {len(df[df['sentiment']=='negative'])}")
    st.write(f"Model Accuracy: {round(acc*100,2)}%")

with st.sidebar.expander("📖 User Guide"):
    st.write("1. Enter a review")
    st.write("2. Click Analyze")
    st.write("3. View sentiment, confidence, and keywords")
    st.write("4. Check charts for analysis")

with st.sidebar.expander("ℹ️ About App"):
    st.info("This app predicts sentiment using ML + rule-based logic with improved confidence scoring.")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze", "📊 Charts", "📜 History", "ℹ️ About"])

# ================= ANALYZE =================
with tab1:
    st.subheader("✍️ Enter Review")

    examples = [
        "Select an example",
        "This movie was fantastic and inspiring",
        "Worst movie ever, waste of time",
        "Amazing acting, but story was weak",
        "I would not recommend this film",
        "Bad movie",
        "It's not a good movie",
        "Somewhat good but not great",
        "Average movie"
    ]

    def update_textarea():
        if st.session_state.example_select != examples[0]:
            st.session_state.review_text = st.session_state.example_select
        else:
            st.session_state.review_text = ""

    st.selectbox("Try Example Reviews", options=examples, key="example_select", on_change=update_textarea)

    review = st.text_area("Enter your review:", key="review_text")

    if st.button("🚀 Analyze"):
        if review.strip() == "":
            st.warning("Please enter a review")
        else:
            cleaned = re.sub(r'[^a-zA-Z ]', ' ', review).lower()
            vec = vectorizer.transform([cleaned])

            # -------- SENTENCE LEVEL --------
            sentences = re.split(r'[.!?]', review)
            sentences = [s.strip() for s in sentences if s.strip()]

            st.subheader("🧠 Sentence Analysis")

            for sent in sentences:
                vec_sent = vectorizer.transform([re.sub(r'[^a-zA-Z ]', ' ', sent).lower()])
                proba = model.predict_proba(vec_sent)[0]
                ml_prediction = model.predict(vec_sent)[0]

                rule = rule_based_fix(sent)

                prediction = rule if rule else ml_prediction

                confidence = round(max(proba)*100, 2)

                # 🔥 IMPROVED BOOST
                if rule:
                    confidence = min(confidence + 20, 100)

                if confidence < 50:
                    prediction = "neutral"

                st.write(f"👉 {sent}")
                st.write(f"➡️ {prediction.capitalize()} ({confidence}%)")
                st.markdown("---")

            # -------- OVERALL --------
            proba_overall = model.predict_proba(vec)[0]
            ml_overall = model.predict(vec)[0]
            overall_rule = rule_based_fix(review)

            overall_pred = overall_rule if overall_rule else ml_overall
            overall_conf = round(max(proba_overall)*100, 2)

            if overall_rule:
                overall_conf = min(overall_conf + 20, 100)

            if overall_conf < 50:
                overall_pred = "neutral"

            st.session_state.history.append({
                "review": review,
                "prediction": overall_pred,
                "confidence": overall_conf
            })

            # -------- DISPLAY --------
            if overall_pred == "positive":
                st.markdown(f'<div class="result-box positive">Positive ({overall_conf}%) 😊</div>', unsafe_allow_html=True)
            elif overall_pred == "negative":
                st.markdown(f'<div class="result-box negative">Negative ({overall_conf}%) 😠</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box neutral">Neutral ({overall_conf}%) 😐</div>', unsafe_allow_html=True)

            st.info(f"📏 Length: {len(review)} characters")

            if show_confidence:
                st.progress(int(overall_conf))

            if show_keywords:
                feature_names = vectorizer.get_feature_names_out()
                top_indices = vec.toarray()[0].argsort()[-5:][::-1]
                st.write("🔑 Keywords:")
                for i in top_indices:
                    st.markdown(f"- **{feature_names[i]}**")

# ================= CHARTS =================
with tab2:
    st.subheader("📊 Charts")
    if st.checkbox("Show Charts"):
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            df.boxplot(column='length', by='sentiment', ax=ax)
            plt.title("Review Length vs Sentiment")
            plt.suptitle("")
            st.pyplot(fig)

        with col2:
            sizes = df['sentiment'].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=sizes.index, autopct='%1.1f%%')
            st.pyplot(fig1)

# ================= HISTORY =================
with tab3:
    st.subheader("📜 History")

    if st.session_state.history:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []

        for item in reversed(st.session_state.history):
            st.markdown(f"**📝 {item['review']}**")
            st.write(f"➡️ {item['prediction'].capitalize()} ({item['confidence']}%)")
            st.markdown("---")

# ================= ABOUT =================
with tab4:
    st.subheader("ℹ️ About Project")
    st.write("""
    This application analyzes movie reviews and predicts sentiment.

    Features:
    - Positive, Negative, Neutral classification
    - Sentence-level analysis
    - Review length analysis
    - Charts & visualization
    - Keyword extraction
    - History tracking
    - Improved confidence scoring (Hybrid ML + Rule)
    """)

# ---------------- FOOTER ----------------
st.markdown("<center>© 2026 Sentiment Analyzer</center>", unsafe_allow_html=True)