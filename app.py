# app.py

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import shap
import numpy as np
from utils import eda
import google.generativeai as genai
import base64
from io import BytesIO
import json

# Configure Gemini API (replace with your API key)
genai.configure(api_key="")  # TODO: Move to secrets in production

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

# =====================
# Load Data & Model
# =====================
data = pd.read_csv("data/banking_data.csv")

with open('models/model.pkl', 'rb') as f:
    saved = pickle.load(f)
    models = saved['models']      # list of (name, model)
    encoders = saved['encoders']
    feature_names = saved['feature_names']
    threshold = saved['threshold']  # loaded dynamically

# Function to convert matplotlib figure to downloadable PNG
def fig_to_download(fig, filename="chart.png"):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">Download {filename}</a>'

# =====================
# Sidebar - Input Form
# =====================
st.sidebar.header("🎯 Customer Input Features")
input_data = {}
month_added = False

for col in feature_names:
    if col in encoders:  # categorical
        input_data[col] = st.sidebar.selectbox(
            f"{col.title()}",
            encoders[col].classes_,
            key=f"{col}_input"
        )
    elif col == 'age':
        input_data['age'] = st.sidebar.slider("Age", 18, 95, 30, key="age_input")
    elif col == 'balance':
        input_data['balance'] = st.sidebar.number_input("Balance", value=0, key="balance_input")
    elif col == 'day':
        input_data['day'] = st.sidebar.slider("Last Contact Day", 1, 31, 15, key="day_input")
    elif col == 'duration':
        input_data['duration'] = st.sidebar.slider("Contact Duration (s)", 0, 3000, 300, key="duration_input")
    elif col == 'campaign':
        input_data['campaign'] = st.sidebar.slider("Campaign Contacts", 1, 50, 1, key="campaign_input")
    elif col == 'pdays':
        input_data['pdays'] = st.sidebar.slider("Days Since Last Contact", -1, 999, -1, key="pdays_input")
    elif col == 'previous':
        input_data['previous'] = st.sidebar.slider("Previous Contacts", 0, 50, 0, key="previous_input")
    elif col in ['month_sin', 'month_cos'] and not month_added:
        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        input_data['month'] = st.sidebar.selectbox(
            "Last Contact Month",
            months,
            key="last_contact_month_input"
        )
        month_added = True

input_df = pd.DataFrame([input_data])

# Apply encoders
for col in encoders:
    input_df[col] = encoders[col].transform(input_df[col])

# Month sin/cos
if 'month_sin' in feature_names and 'month_cos' in feature_names:
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    input_df['month_num'] = month_map[input_data['month']]
    input_df['month_sin'] = np.sin(2 * np.pi * input_df['month_num'] / 12)
    input_df['month_cos'] = np.cos(2 * np.pi * input_df['month_num'] / 12)
    input_df.drop(columns=['month_num'], inplace=True)

# Extra engineered features
if 'balance_per_age' in feature_names:
    input_df['balance_per_age'] = input_df['balance'] / (input_df['age'] + 1)
if 'is_default_and_loan' in feature_names:
    input_df['is_default_and_loan'] = (
        (input_df['default'] == 1) & (input_df['loan'] == 1)
    ).astype(int)

# Reorder
input_df = input_df[feature_names]

# =====================
# Prediction (Auto-run)
# =====================
if st.sidebar.button("🔮 Predict Term Deposit Subscription"):
    avg_proba = np.mean(
        [clf.predict_proba(input_df)[:, 1] for _, clf in models],
        axis=0
    )[0]
    prediction = int(avg_proba >= threshold)

    st.sidebar.success(
        f"Prediction: {'Subscribed ✅' if prediction == 1 else 'Not Subscribed ❌'}\n"
        f"Probability: {avg_proba:.2%}"
    )

    st.session_state['user_input'] = input_df
    st.session_state['probability'] = avg_proba
    st.session_state['prediction'] = prediction
    st.session_state['user_input_data'] = input_data  # Save raw input

    # SHAP for full JSON
    explainer = shap.TreeExplainer(models[0][1])
    shap_vals = explainer.shap_values(input_df)
    # Flatten shap output to (n_features,)
    if isinstance(shap_vals, list):
        vals = np.array(shap_vals[0]).flatten()
    else:
        vals = np.array(shap_vals).flatten()
    all_shap_json = [
        {"feature": feature_names[i], "value": float(vals[i])} for i in range(len(feature_names))
    ]

    final_json = {
        "inputs": input_data,
        "prediction": "Subscribed" if prediction == 1 else "Not Subscribed",
        "probability": float(avg_proba),
        "shap_values": all_shap_json
    }

    st.session_state['full_json'] = final_json

    # Natural-language prompt + JSON for Gemini
    try:
        prompt_nl = (
            f"A customer with probability {avg_proba:.1%} to "
            f"{'subscribe' if prediction == 1 else 'not subscribe'}. "
            "Here is full context JSON:"
        )
        prompt_full = prompt_nl + "\n```json\n" + json.dumps(final_json, indent=2) + "\n```"

        response = model.generate_content(prompt_full)
        st.session_state['gemini_response'] = response.text
    except Exception as e:
        st.session_state['gemini_response'] = f"Could not generate insights: {str(e)}"

# =====================
# Main Tabs
# =====================
tab1, tab2, tab3 = st.tabs(["📊 Historical Data Overview", "🧠 Customer Insights", "💬 AI Insights Chat"])

# ---- Tab 1
with tab1:
    st.subheader("Historical Dataset Trends")

    figs = [
        (eda.plot_age_distribution(data), "age_distribution.png"),
        (eda.plot_job_distribution(data), "job_distribution.png"),
        (eda.plot_marital_status_distribution(data), "marital_status.png"),
        (eda.plot_education_distribution(data), "education_distribution.png"),
        (eda.plot_balance_distribution(data), "balance_distribution.png"),
        (eda.plot_housing_loan_proportion(data), "housing_loan.png"),
        (eda.plot_personal_loan_proportion(data), "personal_loan.png"),
        (eda.plot_term_deposit_subscription(data), "term_deposit.png")
    ]

    for fig, fname in figs:
        st.pyplot(fig)
        st.markdown(fig_to_download(fig, fname), unsafe_allow_html=True)

# ---- Tab 2
with tab2:
    if 'user_input' in st.session_state:
        user_input = st.session_state['user_input']
        user_prob = st.session_state['probability']
        pred = st.session_state['prediction']
        input_data = st.session_state['user_input_data']
        st.markdown(f"### Prediction: {'Subscribed ✅' if pred == 1 else 'Not Subscribed ❌'}")
        st.markdown(f"**Probability:** {user_prob:.2%}")

        # Comparison plots
        fig_age, ax = plt.subplots()
        ax.hist(data['age'], bins=20, color='lightgray', edgecolor='black')
        ax.axvline(input_data['age'], color='red', linestyle='--', label='User Age')
        ax.set_title("Age Distribution with User Highlight")
        ax.legend()
        st.pyplot(fig_age)
        st.markdown(fig_to_download(fig_age, "user_age_comparison.png"), unsafe_allow_html=True)

        fig_balance, ax = plt.subplots()
        ax.hist(data['balance'], bins=20, color='lightgray', edgecolor='black')
        ax.axvline(input_data['balance'], color='red', linestyle='--', label='User Balance')
        ax.set_title("Balance Distribution with User Highlight")
        ax.legend()
        st.pyplot(fig_balance)
        st.markdown(fig_to_download(fig_balance, "user_balance_comparison.png"), unsafe_allow_html=True)

        fig_pie, ax = plt.subplots()
        ax.pie([user_prob, 1 - user_prob], labels=['Subscribed', 'Not Subscribed'], autopct='%1.1f%%')
        ax.set_title("Prediction Probability Breakdown")
        st.pyplot(fig_pie)
        st.markdown(fig_to_download(fig_pie, "probability_breakdown.png"), unsafe_allow_html=True)

        # SHAP waterfall
        explainer = shap.TreeExplainer(models[0][1])
        shap_values = explainer.shap_values(user_input)

        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_val = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
        else:
            base_val = explainer.expected_value

        shap_exp = shap.Explanation(
            values=shap_values[0],
            base_values=base_val,
            data=user_input.iloc[0],
            feature_names=user_input.columns
        )
        fig_shap, ax = plt.subplots()
        shap.waterfall_plot(shap_exp, max_display=10, show=False)
        st.pyplot(fig_shap)
        st.markdown(fig_to_download(fig_shap, "feature_impact.png"), unsafe_allow_html=True)

        # Show JSON
        st.subheader("📄 JSON context sent to LLM")
        st.json(st.session_state['full_json'], expanded=True)
    else:
        st.info("Enter details & click Predict to view insights")

# ---- Tab 3
with tab3:
    if 'user_input' in st.session_state and 'gemini_response' in st.session_state:
        st.subheader("AI-Powered Insights")
        st.write(st.session_state['gemini_response'])

        # Chat mode
        st.subheader("Ask Follow-up Questions")
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        user_question = st.text_input("Ask about this prediction or customer:")
        if user_question:
            try:
                context = (
                    "Using the following JSON context:\n"
                    f"```json\n{json.dumps(st.session_state['full_json'], indent=2)}\n```"
                    f"\nQuestion: {user_question}\n\nAnswer concisely with detailed insights:"
                )
                response = model.generate_content(context)
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("AI", response.text))
            except Exception as e:
                st.error("Error generating response: " + str(e))

        st.subheader("Conversation History")
        for role, txt in st.session_state.chat_history:
            st.markdown(f"**{role}**: {txt}")
            st.markdown("---")
    else:
        st.info("Run a prediction first to unlock chat!")


