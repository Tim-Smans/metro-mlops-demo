import streamlit as st
import requests

# Config
API_URL = "http://localhost/mlflow-model/invocations"
HEADERS = {
    "Content-Type": "application/json",
    "Host": "mlflow-model.local"
}

# Label mapping
label_map = {
    0: "anger",
    1: "joy",
    2: "optimism",
    3: "sadness"
}

# UI
st.set_page_config(page_title="Mood Machine", page_icon="🧠")
st.title("🧠 Mood Machine")
st.subheader("Let AI analyse your mood")

text_input = st.text_area("Enter a sentence:", height=100)

if st.button("Analyse emotions") and text_input.strip() != "":
    payload = {"instances": [text_input.strip()]}

    try:
        response = requests.post(API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        print(response.json())
        pred = response.json()["predictions"][0]
        emotion = label_map.get(pred, "Unkown")

        st.success(f"Feeling: **{emotion.upper()}**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
