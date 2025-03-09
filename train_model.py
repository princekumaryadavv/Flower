import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
file_path = "Iris.csv"
df = pd.read_csv(file_path)

# Drop the 'Id' column
df.drop(columns=['Id'], inplace=True)

# Encode 'Species' into numerical values
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])  # 0 = setosa, 1 = versicolor, 2 = virginica

# Save label encoder for later use
joblib.dump(label_encoder, "label_encoder.pkl")

# Prepare features and labels
X = df.drop(columns=['Species'])
y = df['Species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved successfully!")
