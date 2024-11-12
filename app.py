import streamlit as st
import pandas as pd
import joblib
import mlflow.pyfunc

# Load the ElasticNet model
mlflow.set_tracking_uri("https://dagshub.com/saivardhan4694/datascience_project.mlflow")
model_uri = "mlflow-artifacts:/627b25973d85431787d9911c8e624401/89477567b0b84bca804859448de46630/artifacts/model"
model = mlflow.pyfunc.load_model(model_uri)

# Streamlit App Title
st.title("Wine Quality Prediction App 🍷")
st.write("Enter the characteristics of the wine to predict its quality.")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Function to get user input using Streamlit's columns for better display
def get_user_input():
    fixed_acidity = st.sidebar.number_input("Fixed Acidity", min_value=0.0, max_value=15.0, value=7.4, step=0.1)
    volatile_acidity = st.sidebar.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.7, step=0.01)
    citric_acid = st.sidebar.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    residual_sugar = st.sidebar.number_input("Residual Sugar", min_value=0.0, max_value=20.0, value=1.9, step=0.1)
    chlorides = st.sidebar.number_input("Chlorides", min_value=0.0, max_value=0.1, value=0.076, step=0.001)
    free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=11.0, step=1.0)
    total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=250.0, value=34.0, step=1.0)
    density = st.sidebar.number_input("Density", min_value=0.0, max_value=2.0, value=0.9978, step=0.0001)
    pH = st.sidebar.number_input("pH", min_value=0.0, max_value=10.0, value=3.51, step=0.01)
    sulphates = st.sidebar.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.56, step=0.01)
    alcohol = st.sidebar.number_input("Alcohol", min_value=0.0, max_value=20.0, value=9.4, step=0.1)
    
    # Create a dictionary to hold the input data
    data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    
    # Convert the dictionary to a DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = get_user_input()

# Display user input in a cleaner format
st.markdown("### Selected Wine Features 📊")
for column, value in input_df.iloc[0].items():
    st.markdown(f"**{column.replace('_', ' ').title()}:** {value}")

# Make predictions using the ElasticNet model
if st.button("Predict Quality"):
    prediction = model.predict(input_df)
    st.markdown("### Prediction Result 🍇")
    st.success(f"The predicted wine quality is: **{round(prediction[0], 2)}**")
    st.toast("Prediction complete! 🎉")


# Run the Streamlit app using:
# streamlit run app.py
