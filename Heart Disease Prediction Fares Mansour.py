#Importing some important libaries
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Loading the dataset used
@st.cache_data
def load_data():
    return pd.read_csv("Heart_Disease_Prediction.csv")

# Training and initializing the model
@st.cache_data
def train_model(data):
    data['Heart Disease'] = data['Heart Disease'].replace({'Presence': 1, 'Absence': 0})
    # Splitting data into features and target variable (Heart Disease)
    X = data.drop('Heart Disease', axis=1)
    y = data['Heart Disease']
    
    # Splitting the dataset into training and testing sets for usage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model

# Function to make predictions using the trained model
def predict(model, data):
    prediction = model.predict(data)
    return prediction

# The Main function to use in the web app
def main():
    data = load_data()
    model = train_model(data)
    st.title("Heart Disease Prediction") #this makes the title
    st.sidebar.title("Options") # Sidebar with options 
    option = st.sidebar.selectbox("Select an option", ["Make Prediction"]) #Make prediction the only option made in the sidebar

    # The Make Prediction section
    if option == "Make Prediction":
        st.subheader("Make Prediction") #the title for the subbheader in the sidebar

        # User input form to input all features
        #the values added are random but can be changed
        age = st.number_input("Age", min_value=0, max_value=120, value=44)
        sex = st.radio("Sex (0: Female, 1: Male)", [0, 1], index=1)
        chest_pain_type = st.selectbox("Chest Pain Type", [1, 2, 3, 4], index=2)
        bp = st.number_input("BP", min_value=0, max_value=300, value=130)
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=1000, value=233)
        fbs_over_120 = st.radio("FBS over 120", [0, 1])
        ekg_results = st.selectbox("EKG Results", [0, 1, 2], index=0)
        max_hr = st.number_input("Max HR", min_value=0, max_value=300, value=179)
        exercise_angina = st.radio("Exercise Angina", [0, 1])
        st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=2.5)
        slope_of_st = st.selectbox("Slope of ST", [0, 1, 2], index=2)
        num_vessels_fluro = st.selectbox("Number of Vessels Fluro", [0, 1, 2, 3, 4], index=0)
        thallium = st.selectbox("Thallium", [3, 6, 7], index=2)

        # Make prediction buttons labels
        if st.button("Predict"):
            new_sample = pd.DataFrame({
                'Age': [age],
                'Sex': [sex],
                'Chest pain type': [chest_pain_type],
                'BP': [bp],
                'Cholesterol': [cholesterol],
                'FBS over 120': [fbs_over_120],
                'EKG results': [ekg_results],
                'Max HR': [max_hr],
                'Exercise angina': [exercise_angina],
                'ST depression': [st_depression],
                'Slope of ST': [slope_of_st],
                'Number of vessels fluro': [num_vessels_fluro],
                'Thallium': [thallium]
            })

            # Make predictions function regarding the trained model
            prediction = predict(model, new_sample)
            if prediction[0] == 1:
                st.write("Prediction: Heart Disease (Presence)")
            else:
                st.write("Prediction: No Heart Disease (Absence)")

# Running the app function
if __name__ == '__main__':
    main()