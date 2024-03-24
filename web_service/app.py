from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = Flask(__name__)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = clf.predict(features)[0]
    target_names = iris.target_names[prediction]
    
    return render_template('result.html', prediction=target_names)

if __name__ == '__main__':
    app.run(debug=True)
