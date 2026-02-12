import streamlit as st
import pandas as pd
import joblib
import io
import os

# =====================================================
# PATH CONFIG (IMPORTANT FOR DEPLOYMENT)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "loan_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "Loan_Prediction_model.pkl")

# =====================================================
# LOAD DATA & MODEL
# =====================================================
df = pd.read_csv(DATA_PATH)
Model = joblib.load(MODEL_PATH)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="LoanSphere | Loan Status Predictor",
    layout="wide"
)

st.markdown("<h1 style='text-align:center;'>🏦 LoanSphere – Loan Approval Predictor</h1>", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("ℹ️ About")
    st.write("Predict loan approval using ML model")

    st.metric("Total Records", len(df))

    df["Loan Status"] = df["Loan Status"].map({
        "Rejected": 0,
        "Approved": 1
    })

    st.metric("Approved Loans", int(df["Loan Status"].sum()))
    st.write(f"Model Type: {type(Model).__name__}")

# =====================================================
# TABS
# =====================================================
tab1, tab2 = st.tabs(["🧪 Manual Prediction", "📦 Bulk Prediction"])

# =====================================================
# MANUAL TAB
# =====================================================
with tab1:

    col1, col2 = st.columns(2)

    with col1:
        loan_amount = st.number_input("Current Loan Amount", min_value=0)
        term = st.selectbox("Term", ["Short", "Long"])
        credit_score = st.number_input("Credit Score", 300, 10000, 600)
        annual_income = st.number_input("Annual Income", min_value=0)

    with col2:
        home_ownership = st.selectbox("Home Ownership",
            ["Have Mortgage", "Rent", "Home Mortgage", "Own"])

        purpose = st.selectbox("Purpose",
            ["Business Loan","Buy a Car","Buy House",
             "Debt Consolidation","Educational Expenses",
             "Home Improvements","Major Purchase",
             "Medical Bills","Moving","Other",
             "Renewable Energy","Small Business",
             "Take a Trip","Vacation","Wedding"])

        monthly_debt = st.number_input("Monthly Debt", min_value=0.0)
        credit_history = st.number_input("Years of Credit History", min_value=0)
        months_delinquent = st.number_input("Months since last delinquent", min_value=0)

    open_accounts = st.number_input("Number of Open Accounts", min_value=0)
    credit_problems = st.number_input("Number of Credit Problems", min_value=0)
    current_balance = st.number_input("Current Credit Balance", min_value=0)
    max_credit = st.number_input("Maximum Open Credit", min_value=0)

    # Encoding maps (same as training)
    term_map = {"Short": 0, "Long": 1}
    home_map = {
        "Have Mortgage": 0,
        "Rent": 1,
        "Home Mortgage": 2,
        "Own": 3
    }

    purpose_map = {
        "Business Loan":1,"Buy a Car":2,"Buy House":3,
        "Debt Consolidation":4,"Educational Expenses":5,
        "Home Improvements":6,"Major Purchase":7,
        "Medical Bills":8,"Moving":9,"Other":10,
        "Renewable Energy":11,"Small Business":12,
        "Take a Trip":13,"Vacation":14,"Wedding":15
    }

    input_df = pd.DataFrame([{
        "Current Loan Amount": loan_amount,
        "Term": term_map[term],
        "Credit Score": credit_score,
        "Annual Income": annual_income,
        "Home Ownership": home_map[home_ownership],
        "Purpose": purpose_map[purpose],
        "Monthly Debt": monthly_debt,
        "Years of Credit History": credit_history,
        "Months since last delinquent": months_delinquent,
        "Number of Open Accounts": open_accounts,
        "Number of Credit Problems": credit_problems,
        "Current Credit Balance": current_balance,
        "Maximum Open Credit": max_credit
    }])

    if st.button("🚀 Predict Loan Status"):
        input_df = input_df[Model.feature_names_in_]
        prediction = Model.predict(input_df)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Not Approved")

# =====================================================
# BULK TAB
# =====================================================
with tab2:

    st.subheader("📦 Upload CSV for Bulk Prediction")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            bulk_df = pd.read_csv(uploaded_file)
            st.dataframe(bulk_df.head())

            if st.button("🚀 Run Bulk Prediction"):
                bulk_df_model = bulk_df[Model.feature_names_in_]
                bulk_df["Prediction"] = Model.predict(bulk_df_model)

                st.success("Predictions Completed!")
                st.dataframe(bulk_df.head())

                st.download_button(
                    "📥 Download Results",
                    bulk_df.to_csv(index=False),
                    "loan_predictions.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"Error: {e}")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("<center>🏦 LoanSphere | Powered by Machine Learning</center>", unsafe_allow_html=True)
