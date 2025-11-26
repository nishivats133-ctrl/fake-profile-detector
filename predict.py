import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/rf_pipeline.pkl")

print("\n==========================================")
print("◇ Hybrid AI Fake Profile & Post Detector ◇")
print("==========================================\n")

# --- Take inputs from user ---
followers = int(input("Followers: "))
following = int(input("Following: "))
account_age_days = int(input("Account Age (days): "))
total_posts = int(input("Total Posts: "))
bio = input("Bio (Hindi/English): ")
post_text = input("Post Text (Hindi/English): ")

# --- Create data frame for prediction ---
data = {
    "followers": [followers],
    "following": [following],
    "account_age_days": [account_age_days],
    "total_posts": [total_posts],
    "bio": [bio],
    "post_text": [post_text]
}

X_new = pd.DataFrame(data)

# --- Predict ---
pred = model.predict(X_new)[0]
prob = model.predict_proba(X_new).max()

# --- Rule-based reasoning ---
if "click" in post_text.lower() or "paise" in post_text.lower() or "free" in post_text.lower():
    reason_en = "Post contains suspicious or spam words."
    reason_hi = "पोस्ट में संदिग्ध या स्पैम शब्द हैं।"
    pred = "FAKE"
elif followers < 100 and following > 500:
    reason_en = "Low followers-to-following ratio looks suspicious."
    reason_hi = "फॉलोअर्स और फॉलोइंग का अनुपात संदिग्ध लगता है।"
    pred = "FAKE"
elif account_age_days < 100:
    reason_en = "Newly created account."
    reason_hi = "नया बनाया गया खाता।"
    pred = "FAKE"
else:
    reason_en = "Profile and post appear genuine."
    reason_hi = "प्रोफाइल और पोस्ट वास्तविक लगते हैं।"
    pred = "REAL"  # 🔥 changed this part

# --- Prediction Label in Hindi ---
if pred.lower() == "fake":
    pred_hi = "नकली (FAKE)"
else:
    pred_hi = "वास्तविक (REAL)"

# --- Print Results ---
print("\n-------------------------------")
print(f"Prediction: {pred.upper()} / {pred_hi}")
print(f"Confidence: {prob:.2f}")
print(f"Reason: {reason_en}")
print(f"कारण: {reason_hi}")
print("-------------------------------\n")
