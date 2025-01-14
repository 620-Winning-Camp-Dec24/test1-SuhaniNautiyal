
# AI Tourist Footfall Predictor for Cloud Management

This project aims to predict the tourist footfall for a given location based on factors such as weather, events, and the day of the year. The prediction model is built using a linear regression approach, and the project utilizes a synthetic dataset to simulate real-world conditions. The model can be used for cloud management to optimize resources based on predicted footfall.

## Project Overview

The AI Tourist Footfall Predictor leverages machine learning to predict the number of tourists expected at a particular location. The input features include:
- **Weather**: The weather condition (Sunny, Rainy, or Cloudy).
- **Event**: The type of event (Festival, Holiday, or No Event).
- **Day of Year**: The day of the year (from 1 to 365).

The model is trained on historical data and can predict footfall for any given day based on these features.

## Requirements

Before running the project, make sure you have the following Python libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Code Explanation

### 1. Importing Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')
```

- **pandas** and **numpy** are used for data manipulation and numerical operations.
- **matplotlib** and **seaborn** are used for data visualization.
- **scikit-learn** provides tools for machine learning, including splitting data, creating the linear regression model, and evaluating its performance.
- **joblib** is used for saving the trained model to a file.

### 2. Creating the Dataset

```python
data = {
    'Date': pd.date_range(start='2023-01-01', periods=365, freq='D'),
    'Weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], size=365),
    'Event': np.random.choice(['Festival', 'No Event', 'Holiday'], size=365),
    'Footfall': np.random.randint(1000, 5000, size=365)
}

df = pd.DataFrame(data)
print(df.head())
```

- A synthetic dataset is created to simulate tourist footfall data, with columns for **Date**, **Weather**, **Event**, and **Footfall**.

### 3. Data Preprocessing

```python
df['Weather'] = df['Weather'].map({'Sunny': 2, 'Cloudy': 1, 'Rainy': 0})
df['Event'] = df['Event'].map({'Festival': 2, 'Holiday': 1, 'No Event': 0})
df['DayOfYear'] = df['Date'].dt.dayofyear
df = df.drop('Date', axis=1)

print(df.head())
```

- The **Weather** and **Event** columns are encoded into numerical values.
- A new column **DayOfYear** is created to capture the day of the year.
- The **Date** column is dropped as it's no longer needed.

### 4. Splitting the Data into Training and Testing Sets

```python
X = df.drop('Footfall', axis=1)
y = df['Footfall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
```

- The dataset is split into features (X) and target (y), and further divided into training and testing sets (80% training, 20% testing).

### 5. Training the Model and Evaluating Performance

```python
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
```

- A **Linear Regression** model is trained on the training data, and its performance is evaluated using **Mean Squared Error (MSE)** and **R^2 Score**.

### 6. Making Predictions for New Data

```python
new_data = pd.DataFrame({
    'Weather': [2],  
    'Event': [2],    
    'DayOfYear': [150]  
})
predicted_footfall = model.predict(new_data)
print(f"Predicted Footfall: {int(predicted_footfall[0])}")
```

- The trained model is used to predict the tourist footfall for a given set of conditions (weather, event, day of the year).

### 7. Visualizing Actual vs Predicted Footfall

```python
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.xlabel('Actual Footfall')
plt.ylabel('Predicted Footfall')
plt.title('Actual vs Predicted Footfall')
plt.show()
```

- A scatter plot is created to compare the actual footfall values with the predicted values. This helps in visually assessing the model's accuracy.

### 8. Saving the Model

```python
import joblib

# Save the model to a file
joblib.dump(model, 'footfall_predictor.pkl')
```

- The trained model is saved to a file (`footfall_predictor.pkl`) using **joblib** for future use.

## Conclusion

The **AI Tourist Footfall Predictor** is a machine learning model that predicts the number of tourists based on weather, events, and the day of the year. The model uses linear regression and can be easily adapted for real-world applications such as cloud resource management, event planning, or tourism analytics.
