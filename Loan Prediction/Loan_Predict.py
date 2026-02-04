import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Loan Status Predictor")
st.title("🏦 Loan Status Prediction")
st.write("Enter applicant details to predict loan approval status")

# -----------------------------
# LOAD DATA
# -----------------------------
data_path = r"E:\BA BI\Project BBC\project2\Project\loan_data.csv"
df = pd.read_csv(data_path)

# -----------------------------
# LOAD MODEL
# -----------------------------
model_path = r"E:\BA BI\Project BBC\project2\Project\Loan_Prediction_model.pkl"
with open(model_path, "rb") as file:
    Model = pickle.load(file)

st.success("Model loaded successfully!")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app predicts **Loan Status**
    (1 = Approved, 0 = Not Approved)
    """)

    st.header("📊 Dataset Info")
    st.metric("Total Records", len(df))
    df["Loan Status"] = df["Loan Status"].map({
    "Rejected": 0,
    "Approved": 1
})

    st.metric("Loan Approved", int(df["Loan Status"].sum()))


    st.header("🔧 Model Info")
    st.write(f"Model Type: {type(Model).__name__}")

# -----------------------------
# USER INPUTS
# -----------------------------
st.markdown("---")
st.subheader("📋 Applicant Details")

loan_amount = st.number_input("Current Loan Amount", min_value=0)
term_map = {"Short": 0, "Long": 1}
term_label = st.selectbox("Term", list(term_map.keys()))
term = term_map[term_label]
credit_score = st.number_input("Credit Score", min_value=300, max_value=10000, value=600)
annual_income = st.number_input("Annual Income", min_value=0)

home_ownership_map = {
    "Have Mortgage": 0,
    "Rent": 1,
    "Home Mortgage": 2,
    "Own": 3
}

home_ownership_label = st.selectbox(
    "Home Ownership",
    list(home_ownership_map.keys())
)

home_ownership = home_ownership_map[home_ownership_label]


purpose_map = {
    "Business Loan": 1,
    "Buy a Car": 2,
    "Buy House": 3,
    "Debt Consolidation": 4,
    "Educational Expenses": 5,
    "Home Improvements": 6,
    "Major Purchase": 7,
    "Medical Bills": 8,
    "Moving": 9,
    "Other": 10,
    "Renewable Energy": 11,
    "Small Business": 12,
    "Take a Trip": 13,
    "Vacation": 14,
    "Wedding": 15
}

purpose_label = st.selectbox(
    "Loan Purpose",
    list(purpose_map.keys())
)

purpose = purpose_map[purpose_label]

monthly_debt = st.number_input("Monthly Debt", min_value=0.0)
credit_history = st.slider("Years of Credit History", 300, 8000, 1000)
months_delinquent = st.number_input("Months Since Last Delinquent", min_value=0)
open_accounts = st.number_input("Number of Open Accounts", min_value=0)
credit_problems = st.number_input("Number of Credit Problems", min_value=0)
current_balance = st.number_input("Current Credit Balance", min_value=0)
max_credit = st.number_input("Maximum Open Credit", min_value=0)

# -----------------------------
# CREATE INPUT DATAFRAME
# -----------------------------
input_df = pd.DataFrame([{
    "Current Loan Amount": loan_amount,
    "Term": term,
    "Credit Score": credit_score,
    "Annual Income": annual_income,
    "Home Ownership": home_ownership,
    "Purpose": purpose,
    "Monthly Debt": monthly_debt,
    "Years of Credit History": credit_history,
    "Months since last delinquent": months_delinquent,
    "Number of Open Accounts": open_accounts,
    "Number of Credit Problems": credit_problems,
    "Current Credit Balance": current_balance,
    "Maximum Open Credit": max_credit
}])

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🚀 Predict Loan Status"):
    
    input_df = input_df[Model.feature_names_in_]
    prediction = Model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
