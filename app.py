# Import required libraries
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Create a Flask app
app = Flask(__name__)

# Load the trained model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Read the train data file
train_data = pd.read_csv('train.csv')

# Clean data by dropping rows with null values
train_data = train_data.dropna()

# Drop unnecessary columns
loanStatus = train_data['Loan_Status']
train_data = train_data.drop(['Loan_ID', 'Loan_Status'], axis=1)

# Define categorical columns
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History', 'Loan_Amount_Term']

# Initialize the OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')

# Use OneHotEncoder to transform the categorical columns
ohe_X = pd.DataFrame(encoder.fit_transform(train_data[categorical_columns]))

# Assign column names to one-hot encoded DataFrame
ohe_X.columns = encoder.get_feature_names_out()

# Drop original categorical columns from the 'train_data' DataFrame
train_data.drop(categorical_columns, axis=1, inplace=True)

# Set the index of the one-hot encoded DataFrame to match 'train_data' index
ohe_X.index = train_data.index

# Concatenate the one-hot encoded DataFrame with remaining columns
train_data = pd.concat([train_data, ohe_X], axis=1)

# Prepare features and target variable
X = train_data
y = loanStatus

# Fit the model to the training data
model.fit(X, y)

# Define a route for loan status prediction
@app.route('/predict_loan_status', methods=['POST'])
def predict_loan_status():
    if request.method == 'POST':
        # Get input data as JSON
        input_data = request.get_json()
        
        # Prepare input data for prediction
        input_df = pd.DataFrame([input_data])
        input_df = input_df.drop(['Loan_Status'], axis=1)
        
        # Perform one-hot encoding
        input_df = pd.DataFrame(encoder.transform(input_df[categorical_columns]))
        input_df.columns = encoder.get_feature_names_out(categorical_columns)
        
        # Drop original categorical columns
        input_df.index = [0]
        
        # Concatenate input data
        input_data = pd.concat([input_df, input_data], axis=1)
        
        # Make predictions
        prediction = model.predict(input_data)
        
        return jsonify({'Loan_Status': prediction[0]})

# Define a route to render a form for user input
@app.route('/')
def loan_status_form():
    return render_template('loan_status_form.html')

if __name__ == '__main__':
    app.run(debug=True)
