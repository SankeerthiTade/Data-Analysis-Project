# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the Dataset
data = pd.read_csv(r'C:\Users\Sankeerthi\Downloads\sample_battery_data.csv')

# Step 3: Data Analysis
# Inspect the data
print(data.head())
print(data.info())
print(data.describe())

# Check for missing values
print("Missing Values:\n", data.isnull().sum())

# Visualize the distribution of features with adjusted settings
pairplot = sns.pairplot(data)
for ax in pairplot.axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')

plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Additional Graphs

# Step 3A: Distribution of Effective SOC
plt.figure(figsize=(8, 5))
sns.histplot(data['Effective SOC'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Effective SOC')
plt.xlabel('Effective SOC')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Step 3B: Boxplot for Battery Voltages
plt.figure(figsize=(10, 5))
sns.boxplot(data=data[['Fixed Battery Voltage', 'Portable Battery Voltage']], palette='Set2')
plt.title('Boxplot of Battery Voltages')
plt.ylabel('Voltage')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Step 4: Feature Selection
# Define features and target variable
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

# Step 5: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Machine Learning Model Development
# Initialize and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# Step 8: Define and Calculate KPIs
def calculate_kpi(y_true, y_pred):
    # Example KPI calculations
    charge_cycles = np.mean(y_pred)  # Example calculation (this should be based on domain knowledge)
    range_kpi = np.max(y_pred) - np.min(y_pred)  # Example KPI for range
    battery_performance = np.mean(y_pred) / np.mean(y_true)  # Performance comparison
    return {
        'Charge Cycle': charge_cycles,
        'Range': range_kpi,
        'Battery Performance': battery_performance
    }

kpis = calculate_kpi(y_test, y_pred)
print('KPIs:', kpis)

# Step 9: Documentation and Presentation
# Create a report (for simplicity, we'll print to console; consider using Jupyter Notebook or saving to file)
report = f"""
Data Analysis Report
----------------------
- Mean Squared Error: {mse}
- R² Score: {r2}
- KPIs:
    - Charge Cycle: {kpis['Charge Cycle']}
    - Range: {kpis['Range']}
    - Battery Performance: {kpis['Battery Performance']}
"""

print(report)

# Save the report to a text file
with open(r'C:\Users\Sankeerthi\Downloads\data_analysis_report.txt', 'w') as f:
    f.write(report)
