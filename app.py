import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib  # for saving/loading model
import numpy as np

# --- Load or train your model and scaler here ---

@st.cache_resource
def load_model_and_scaler():
    # For demo, we simulate loading from disk or training
    
    # You would normally load like:
    # model = joblib.load('rf_model.pkl')
    # scaler = joblib.load('scaler.pkl')
    
    # For demo, we train a dummy model on sample data here
    df = pd.read_csv('election_data.csv')
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Result (Pass/Fail)'])
    df['Amount of Bond/Tax'] = df['Amount of Bond/Tax'].astype(str).str.replace(r'[\$,]', '', regex=True)
    df['Amount of Bond/Tax'] = pd.to_numeric(df['Amount of Bond/Tax'], errors='coerce')
    df['Amount of Bond/Tax'] = df['Amount of Bond/Tax'].fillna(df['Amount of Bond/Tax'].mean())
    df['% Yes'] = pd.to_numeric(df['% Yes'], errors='coerce').fillna(df['% Yes'].mean())
    df['% No'] = pd.to_numeric(df['% No'], errors='coerce').fillna(df['% No'].mean())
    df['Result'] = df['Result (Pass/Fail)'].str.lower().map({'pass': 1, 'fail': 0})

    y = df['Result']
    cols_to_drop = ['Result', 'Result (Pass/Fail)', 'Election Date', 'Threshold', 'Election Year']
    X = df.drop(columns=cols_to_drop)
    categorical_cols = ['Agency County', 'Agency Name', 'Type of Tax/Debt', 'Purpose', 'Measure', 'Election Type']
    for col in categorical_cols:
        X[col] = X[col].astype('category')
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, X.columns

# Load model, scaler, and feature columns once
model, scaler, feature_cols = load_model_and_scaler()

st.title("Election Outcome Prediction")

# Sidebar inputs for user to enter features
st.sidebar.header("Input Election Features")

def user_input_features():
    # For simplicity, only a few inputs here; extend as needed

    amount_of_bond = st.sidebar.number_input('Amount of Bond/Tax', min_value=0, value=1000000)
    percent_yes = st.sidebar.slider('% Yes', 0, 100, 50)
    percent_no = st.sidebar.slider('% No', 0, 100, 50)
    
    # Example categorical input - you can add more categories based on your data
    agency_county = st.sidebar.selectbox('Agency County', ['County A', 'County B', 'County C'])
    election_type = st.sidebar.selectbox('Election Type', ['General', 'Primary', 'Special'])
    
    # Put inputs in a dict
    data = {
        'Amount of Bond/Tax': amount_of_bond,
        '% Yes': percent_yes,
        '% No': percent_no,
        # For categories, we'll one-hot encode after
        'Agency County': agency_county,
        'Election Type': election_type,
        # Fill other categorical columns with default or empty
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Preprocess input_df to match training features (one-hot encoding)

def preprocess_input(df_input):
    # Start with zeros for all columns used in training
    X_new = pd.DataFrame(0, index=[0], columns=feature_cols)

    # Set numeric features
    for col in ['Amount of Bond/Tax', '% Yes', '% No']:
        if col in X_new.columns:
            X_new.loc[0, col] = df_input.loc[0, col]

    # Set categorical features one-hot encoded (they have column names like Agency County_XYZ)
    for col in df_input.columns:
        if col not in ['Amount of Bond/Tax', '% Yes', '% No']:
            cat_col_prefix = f"{col}_"
            for feat_col in feature_cols:
                if feat_col.startswith(cat_col_prefix) and feat_col.endswith(df_input.loc[0, col]):
                    X_new.loc[0, feat_col] = 1
                    break
    return X_new

X_user = preprocess_input(input_df)

# Scale user input
X_user_scaled = scaler.transform(X_user)

# Predict button
if st.button('Predict Election Result'):
    pred = model.predict(X_user_scaled)[0]
    proba = model.predict_proba(X_user_scaled)[0]

    result = "Pass" if pred == 1 else "Fail"
    st.write(f"### Prediction: {result}")
    st.write(f"Probability Pass: {proba[1]:.2f}")
    st.write(f"Probability Fail: {proba[0]:.2f}")
