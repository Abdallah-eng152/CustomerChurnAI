
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="📊",
    layout="wide"
)


st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0B1120;
        color: #E2E8F0;
    }

    /* Main content width and spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Headings */
    h1, h2, h3 {
        color: #F1F5F9 !important;
        font-weight: 700 !important;
    }

    /* Normal text */
    p, label, .stMarkdown {
        color: #CBD5E1;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #111C33, #162442);
        border: 1px solid #263B60;
        padding: 20px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #38BDF8 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #2563EB, #0EA5E9);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-weight: 600;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        border: 1px solid #7DD3FC;
        color: white;
        filter: brightness(1.12);
    }

    /* Input fields */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #111C33;
        border-color: #334155;
        border-radius: 8px;
    }

    /* Dividers */
    hr {
        border-color: #263B60;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #080D18;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("logistic_regression_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Run main.py first to save the model."
    )
    st.stop()





st.markdown("""
<div style="background-color:#111C33; border:1px solid #263B60; border-radius:18px; padding:28px 30px; margin-bottom:25px;">
    <p style="color:#38BDF8; font-size:13px; font-weight:bold; letter-spacing:3px;">
        MACHINE LEARNING PROJECT
    </p>
    <h1 style="color:#F8FAFC; font-size:36px;">
        Customer Churn AI
    </h1>
    <p style="color:#94A3B8; font-size:16px;">
        Analyze customer behavior and predict churn risk using machine learning.
    </p>
    <p style="color:#4ADE80; font-weight:bold;">
        ● MODEL DASHBOARD
    </p>
</div>
""", unsafe_allow_html=True)

st.write("HEADER CHECK")


@st.cache_data
def load_data():
    df = pd.read_csv("data/Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )
    df = df.dropna(subset=["TotalCharges"])
    return df

df = load_data()

st.divider()
st.header("📈 Customer Churn Dashboard")

total_customers = len(df)
churned_customers = (df["Churn"] == "Yes").sum()
churn_rate = churned_customers / total_customers

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Customers Churned", f"{churned_customers:,}")
col3.metric("Churn Rate", f"{churn_rate:.1%}")

st.subheader("Churn by Contract Type")

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

st.bar_chart(contract_churn)

st.subheader("Churn by Tenure")

df["Churn Status"] = df["Churn"].map({
    "Yes": "Churned",
    "No": "Stayed"
})

st.line_chart(
    df.groupby("tenure")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .rename("Churn Rate (%)")
)

st.caption(
    "Dashboard statistics are calculated from the cleaned dataset."
)

st.divider()

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=72, value=12
    )
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

with col2:
    internet = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

with col3:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    paperless = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, value=70.0
    )
    total_charges = st.number_input(
        "Total Charges", min_value=0.0, value=840.0
    )

st.divider()

if st.button("Predict Churn", type="primary", use_container_width=True):
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    probability = model.predict_proba(customer)[0, 1]
    threshold = 0.40

    st.subheader("Prediction Result")
    st.metric("Estimated Churn Probability", f"{probability:.1%}")
    st.progress(float(probability))

    if probability >= threshold:
        st.error("High Churn Risk: This customer may leave the service.")
    else:
        st.success("Low Churn Risk: This customer is less likely to leave.")

    st.caption(
        "This is a machine-learning estimate, not a guarantee. "
        "The decision threshold is set to 0.40."
    )

    
st.divider()
st.header("🤖 Model Performance Comparison")

performance = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy (%)": [80.38, 76.76],
    "Churn Recall (%)": [57, 65]
})

st.dataframe(performance, hide_index=True, use_container_width=True)

st.bar_chart(
    performance.set_index("Model")[["Accuracy (%)", "Churn Recall (%)"]]
)

st.caption(
    "Accuracy measures overall correctness. Churn recall measures "
    "the percentage of customers who actually churned and were identified."
)