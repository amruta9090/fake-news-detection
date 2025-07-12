
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load datasets
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Label data
true_df["label"] = 1
fake_df["label"] = 0

# Combine title and text
true_df["text"] = true_df["title"] + " " + true_df["text"]
fake_df["text"] = fake_df["title"] + " " + fake_df["text"]

# Merge and shuffle
df = pd.concat([true_df, fake_df])
df = df[["text", "label"]].sample(frac=1).reset_index(drop=True)

# Vectorization
X = df["text"]
y = df["label"]
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Streamlit App
st.title("📰 Fake News Detection using AI/ML")
st.subheader("Enter a news headline or paragraph to check if it's Real or Fake")

user_input = st.text_area("📝 News Text", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        vec = vectorizer.transform([user_input])
        prob = model.predict_proba(vec)[0]
        pred = model.predict(vec)[0]

        fake_prob = prob[0]
        real_prob = prob[1]

        if pred == 0:
            st.error(f"🟥 Prediction: Fake News")
            st.write(f"📊 Risk Score: {int(fake_prob * 100)} / 100")
        else:
            st.success(f"✅ Prediction: Real News")
            st.write(f"📊 Confidence Score: {int(real_prob * 100)} / 100")

        # Show top keywords
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_sorting = np.argsort(vec.toarray()).flatten()[::-1]
        top_words = feature_array[tfidf_sorting][:5]

        st.markdown("**🔍 Top Influential Keywords:**")
        for word in top_words:
            st.write(f"• {word}")
