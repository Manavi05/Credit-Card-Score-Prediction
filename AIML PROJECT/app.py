from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

# Function to plot input feature values
def plot_input_features(input_data):
    """
    Plots the input feature values as a bar chart.
    """
    plt.figure(figsize=(8, 5))
    input_data.T.plot(kind='bar', legend=False, color='blue')
    plt.title("Input Feature Values")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/input_graph.png')  # Save the plot to the static folder
    plt.close()

# Function to plot feature importance
def plot_feature_importance(feature_names, importance_values):
    """
    Plots the feature importances of the model.
    """
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importance_values, color='green')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('static/feature_importance_plot.png')  # Save the plot
    plt.close()

# Function to plot prediction distribution
def plot_prediction_distribution(predictions):
    """
    Plots the distribution of predictions.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(predictions, bins=10, color='orange', edgecolor='black')
    plt.xlabel('Predicted Classes')
    plt.ylabel('Frequency')
    plt.title('Prediction Distribution')
    plt.tight_layout()
    plt.savefig('static/prediction_distribution_plot.png')  # Save the plot
    plt.close()

# Load the trained model
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file) 

# Home route for displaying the input form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get input values from the form
            feature1 = float(request.form['DerogCnt'])
            feature2 = float(request.form['InqCnt06'])
            feature3 = float(request.form['TLMaxSum'])
            feature4 = float(request.form['TL75UtilCnt'])

            # Prepare the input data
            input_data = pd.DataFrame([[feature1, feature2, feature3, feature4]],
                                      columns=['DerogCnt', 'InqCnt06', 'TLMaxSum', 'TL75UtilCnt'])
            
            # Generate input feature plot
            plot_input_features(input_data)

            # Generate feature importance plot (if supported by the model)
            if hasattr(model, "feature_importances_"):
                feature_names = ['DerogCnt', 'InqCnt06', 'TLMaxSum', 'TL75UtilCnt']
                feature_importances = model.feature_importances_
                plot_feature_importance(feature_names, feature_importances)

            # Generate prediction and distribution plot
            prediction = model.predict(input_data)[0]
            predictions = model.predict(input_data)
            plot_prediction_distribution(predictions)

            # Render the result template with the prediction
            return render_template('result.html', prediction=prediction)
        except ValueError as e:
            return f"Error: {e}. Please make sure all inputs are valid numbers."
    return render_template('index.html')

# Run the Flask app
if __name__ == "__main__":
   import os
port = int(os.environ.get('PORT', 5000))
app.run(debug=True, host='0.0.0.0', port=port)

