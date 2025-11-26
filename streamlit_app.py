import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/rf_pipeline.pkl")

st.title("🔍 Hybrid AI Fake Profile & Post Detector (Hindi + English)")

followers = st.number_input("Followers:", min_value=0)
following = st.number_input("Following:", min_value=0)
account_age_days = st.number_input("Account age (days):", min_value=0)
total_posts = st.number_input("Total posts:", min_value=0)
bio = st.text_input("Bio (Hindi/English):")
post_text = st.text_area("Post Text (Hindi/English):")

if st.button("Predict"):
    data = {
        "followers": [followers],
        "following": [following],
        "account_age_days": [account_age_days],
        "total_posts": [total_posts],
        "bio": [bio],
        "post_text": [post_text]
    }
    df = pd.DataFrame(data)
    pred = model.predict(df)[0]
    prob = model.predict_proba(df).max()

    st.write("### Result:")
    st.write(f"**Prediction:** {pred.upper()}")
    st.write(f"**Confidence:** {prob:.2f}")

    if "click" in post_text.lower() or "free" in post_text.lower():
        reason = "Post contains suspicious/sCam keywords."
    elif followers < 100 and following > 500:
        reason = "Low follower ratio – suspicious."
    elif account_age_days < 30:
        reason = "Very new account – possibly fake."
    else:
        reason = "Profile seems genuine."

    st.write("**Reason:**", reason)
