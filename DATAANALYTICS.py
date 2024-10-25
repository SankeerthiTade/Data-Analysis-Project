# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Step 1: Set the title of the Streamlit app
st.title("Battery Data Analysis and Machine Learning")

# Step 2: Load the Dataset
data = pd.read_csv(r'C:\Users\Sankeerthi\Downloads\sample_battery_data.csv')

# Step 3: Data Analysis
# Display the dataset
st.subheader("Dataset Overview")
st.write(data.head())

# Display information about the dataset
st.subheader("Data Info")
st.text(data.info())  # Display info as text
st.write("Data Types:")
st.write(data.dtypes)  # Display data types
st.write("Missing Values:")
st.write(data.isnull().sum())  # Display missing values count

# Step 4: Visualizations
st.subheader("Pairplot of Features")
if st.button("Show Pairplot"):
    pairplot = sns.pairplot(data)
    st.pyplot(pairplot)

# Correlation matrix
st.subheader("Correlation Matrix")
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
st.pyplot(plt)

# Additional Graphs
st.subheader("Distribution of Effective SOC")
plt.figure(figsize=(8, 5))
sns.histplot(data['Effective SOC'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Effective SOC')
plt.xlabel('Effective SOC')
plt.ylabel('Frequency')
plt.grid(True)
st.pyplot(plt)

st.subheader("Boxplot for Battery Voltages")
plt.figure(figsize=(10, 5))
sns.boxplot(data=data[['Fixed Battery Voltage', 'Portable Battery Voltage']], palette='Set2')
plt.title('Boxplot of Battery Voltages')
plt.ylabel('Voltage')
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(plt)

# Step 5: Feature Selection
features = [
    'Fixed Battery Voltage',
    'Portable Battery Voltage',
    'Portable Battery Current',
    'Fixed Battery Current',
    'Motor Status (On/Off)',
    'BCM Battery Selected',
    'Portable Battery Temperatures',
    'Fixed Battery Temperatures'
]
target = 'Effective SOC'

X = data[features]
y = data[target]

# Step 6: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Machine Learning Model Development
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display model evaluation results
st.subheader("Model Evaluation Results")
st.write(f'Mean Squared Error: {mse}')
st.write(f'R² Score: {r2}')

# Step 9: Define and Calculate KPIs
def calculate_kpi(y_true, y_pred):
    charge_cycles = np.mean(y_pred)
    range_kpi = np.max(y_pred) - np.min(y_pred)
    battery_performance = np.mean(y_pred) / np.mean(y_true)
    return {
        'Charge Cycle': charge_cycles,
        'Range': range_kpi,
        'Battery Performance': battery_performance
    }

kpis = calculate_kpi(y_test, y_pred)

# Display KPIs
st.subheader("Key Performance Indicators (KPIs)")
st.write(f'Charge Cycle: {kpis["Charge Cycle"]}')
st.write(f'Range: {kpis["Range"]}')
st.write(f'Battery Performance: {kpis["Battery Performance"]}')
