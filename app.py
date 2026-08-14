import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('titanic_pipeline_model.pkl')

st.title("🚢 Titanic Survival Prediction App")
st.write("Will you survive the Titanic disaster? Enter your details below to find out!")

# Create input fields for the user
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.slider("Fare Paid (£)", 0.0, 500.0, 32.2)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Calculate engineered feature used in pipeline
family_size = sibsp + parch + 1

if st.button("Predict Survival"):
    # Create a dataframe for the pipeline
    user_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'Fare': [fare],
        'Embarked': [embarked],
        'FamilySize': [family_size]
    })
    
    # Make prediction
    prediction = model.predict(user_data)[0]
    
    if prediction == 1:
        st.success("🎉 Prediction: You would have Survived!")
        st.balloons()
    else:
        st.error("💀 Prediction: Unfortunately, you did not survive.")
