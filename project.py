import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns



# Read the CSV file with the first row as column names
df = pd.read_csv("cardio_train.csv", sep=';', header=0)



X = df.iloc[:, :-1]  # Features are all columns except the last one
y = df.iloc[:, -1]   # Target column is the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


# Plotting the predicted vs. actual values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('RandomForest Classifier: Predicted vs. Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


# Set the style for Seaborn plots (optional)
sns.set(style='whitegrid')

# Pairplot for numerical columns
sns.pairplot(df[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol']], diag_kind='kde')
plt.title('Pairplot of Numerical Columns')
plt.show()

# Distribution of 'age' by 'cardio' (target variable)
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='age', hue='cardio', kde=True)
plt.title('Distribution of Age by Cardiovascular Disease')
plt.show()

# Boxplot showing 'cholesterol' levels by 'gender'
plt.figure(figsize=(8, 6))
sns.boxplot(x='cholesterol', y='gender', data=df)
plt.title('Cholesterol Levels by Gender')
plt.show()

# Scatterplot of 'weight' vs 'height'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='weight', y='height', data=df, hue='cardio')
plt.title('Weight vs Height')
plt.show()

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Features')
plt.show()