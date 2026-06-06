import streamlit as st
import pandas as pd
import pickle


# =========================
# Load Files
# =========================

models = pickle.load(
    open("loan_model.pkl","rb")
)

model = models["Logistic Regression"]


scaler = pickle.load(
    open("scaler.pkl","rb")
)

encoder = pickle.load(
    open("encoder.pkl","rb")
)



# =========================
# UI
# =========================


st.title("🏦 Loan Approval Prediction System")



# =========================
# Categorical Columns
# SAME AS TRAINING
# =========================


Employer_Category = st.selectbox(
    "Employer Category",
    [
        "Government",
        "MNC",
        "Private",
        "Unemployed"
    ]
)


Gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)


Property_Area = st.selectbox(
    "Property Area",
    [
        "Rural",
        "Semiurban",
        "Urban"
    ]
)


Loan_Purpose = st.selectbox(
    "Loan Purpose",
    [
        "Car",
        "Education",
        "Home",
        "Personal"
    ]
)


Employment_Status = st.selectbox(
    "Employment Status",
    [
        "Salaried",
        "Self-employed",
        "Unemployed"
    ]
)


Marital_Status = st.selectbox(
    "Marital Status",
    [
        "Married",
        "Single"
    ]
)



# =========================
# Numerical Columns
# SAME ORDER
# =========================


Applicant_Income = st.number_input(
    "Applicant Income",
    min_value=0.0
)


Coapplicant_Income = st.number_input(
    "Coapplicant Income",
    min_value=0.0
)


Age = st.number_input(
    "Age",
    min_value=18.0
)


Dependents = st.number_input(
    "Dependents",
    min_value=0.0
)


Credit_Score = st.number_input(
    "Credit Score",
    min_value=300.0,
    max_value=900.0
)


Existing_Loans = st.number_input(
    "Existing Loans",
    min_value=0.0
)


DTI_Ratio = st.number_input(
    "DTI Ratio",
    min_value=0.0
)


Savings = st.number_input(
    "Savings",
    min_value=0.0
)


Collateral_Value = st.number_input(
    "Collateral Value",
    min_value=0.0
)


Loan_Amount = st.number_input(
    "Loan Amount",
    min_value=0.0
)


Loan_Term = st.number_input(
    "Loan Term",
    min_value=1.0
)


Education_Level = st.selectbox(
    "Education Level",
    [
        "Graduate",
        "Not Graduate"
    ]
)



# =========================
# Prediction
# =========================


if st.button("Check Loan Eligibility"):



    # ---------- Encoder ----------

    cat_df = pd.DataFrame(
        [[
            Employer_Category,
            Gender,
            Property_Area,
            Loan_Purpose,
            Employment_Status,
            Marital_Status
        ]],
        columns=encoder.feature_names_in_
    )


    encoded_array = encoder.transform(
        cat_df
    )


    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out()
    )



    # ---------- Label Encoder ----------

    if Education_Level == "Graduate":
        Education_Level = 0

    else:
        Education_Level = 1



    # ---------- Numerical ----------


    num_df = pd.DataFrame(
        [[
            Applicant_Income,
            Coapplicant_Income,
            Age,
            Dependents,
            Credit_Score,
            Existing_Loans,
            DTI_Ratio,
            Savings,
            Collateral_Value,
            Loan_Amount,
            Loan_Term,
            Education_Level
        ]],

        columns=[
            "Applicant_Income",
            "Coapplicant_Income",
            "Age",
            "Dependents",
            "Credit_Score",
            "Existing_Loans",
            "DTI_Ratio",
            "Savings",
            "Collateral_Value",
            "Loan_Amount",
            "Loan_Term",
            "Education_Level"
        ]
    )



    # =========================
    # FINAL ORDER
    #
    # OneHotEncoder
    #       +
    # Numerical
    # =========================


    final_data = pd.concat(
        [
            encoded_df,
            num_df
        ],
        axis=1
    )
    

    final_data = final_data[scaler.feature_names_in_ ]
    

    final_scaled = scaler.transform( final_data)
    
    prediction = model.predict( final_scaled)
    

    
    st.write("Model Output:",prediction[0])
    

    if prediction[0] == 1:
        
        st.success("✅ Loan Approved")
        
    else:
        st.error("❌ Loan Rejected")

