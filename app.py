import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from joblib import load

# Create a Flask app
app = Flask(__name__)

# Load the trained model
model = load('loan_model.joblib')

# Read the train data file
train_data = pd.read_csv('train.csv')

# Clean data by dropping rows with null values
train_data = train_data.dropna()

# Drop unnecessary columns
loanStatus = train_data['Loan_Status']
train_data = train_data.drop(['Loan_ID', 'Loan_Status'], axis=1)

categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History', 'Loan_Amount_Term']

# Initialize the OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# Use OneHotEncoder to transform the categorical columns
ohe_X = pd.DataFrame(encoder.fit_transform(train_data[categorical_columns]))

# Assign column names to one-hot encoded DataFrame
encoded_columns = encoder.get_feature_names_out(categorical_columns)
ohe_X.columns = encoded_columns

# Drop original categorical columns from the 'train_data' DataFrame
train_data.drop(categorical_columns, axis=1, inplace=True)

# Set the index of the one-hot encoded DataFrame to match 'train_data' index
ohe_X.index = train_data.index

# Concatenate the one-hot encoded DataFrame with remaining columns
train_data = pd.concat([train_data, ohe_X], axis=1)

# Define categorical columns after encoding
categorical_columns = encoded_columns.tolist()


# Prepare features and target variable
X = train_data
Y = loanStatus

try:
    sample_data = X.head(1)  # Take a sample data point from the prepared data
    prediction = model.predict(sample_data)
    print("Model loaded successfully!")
    print("Prediction:", prediction)
except Exception as e:
    print("Error loading the model:", e)

# Define a route for loan status prediction
@app.route('/predict_loan_status', methods=['POST'])
def predict_loan_status():
    if request.method == 'POST':
        # Get input data as JSON
        input_data = request.get_json()

        # Print the input data for debugging
        print("\n \n Input Data:\n")
        print(input_data)


        first_key = True  # Flag to identify the first key
        for key in input_data:
            if first_key:
                first_key = False
            else:
                if isinstance(input_data[key], int):
                    input_data[key] = float(input_data[key])


        # Prepare input data for prediction
        input_df = pd.DataFrame([input_data])

        print("\nInput Dataframe:")
        print(input_df)

        # Ensure the input data columns match the model's columns
        if not input_df.columns.equals(X.columns):
            return jsonify({'error': 'Input data columns do not match the model'})
        else:
            print("Input data columns match the model.")

        # Extract the categorical columns from the input data
        input_categorical = input_df[categorical_columns]

        # Ensure the input data's categorical columns match the model's
        if not input_categorical.columns.equals(categorical_columns):
            return jsonify({'error': 'Input data categorical columns do not match the model'})
        else:
            print("Categorical columns match the model.")

        # Perform one-hot encoding on the input data's categorical columns
        input_encoded = pd.DataFrame(encoder.transform(input_categorical))
        input_encoded.columns = encoder.get_feature_names_out(categorical_columns)

        # Drop original categorical columns
        input_df = input_df.drop(categorical_columns, axis=1)

        # Concatenate the one-hot encoded categorical columns with the remaining input data
        input_df = pd.concat([input_df, input_encoded], axis=1)

        # Make predictions
        prediction = model.predict(input_df)

        # Print the prediction for debugging
        print("Prediction:")
        print(prediction)

        # Return the prediction result in JSON format
        return jsonify({'Loan_Status': prediction[0]})




# Define a route to render the form for user input
@app.route('/')
def loan_status_form():
    return render_template('loan_status_form.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)

