import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data using pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Preprocess data to ensure consistency in shape and type
max_features = max(len(sample) for sample in data_dict['data'])
processed_data = np.zeros((len(data_dict['data']), max_features))

for i, sample in enumerate(data_dict['data']):
    processed_data[i, :len(sample)] = sample  # Assuming you want to pad shorter samples with zeros

# Convert processed data and labels into NumPy arrays
data = np.asarray(processed_data)
labels = np.asarray(data_dict['labels'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the RandomForestClassifier model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)

# Print the accuracy score
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
