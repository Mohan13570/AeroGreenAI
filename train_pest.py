import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset_path = "pest_dataset"

classes = ["no_pest", "pest"]

X = []
y = []

for label, category in enumerate(classes):
    folder = os.path.join(dataset_path, category)

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128,128))

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            features = hsv.flatten()

            X.append(features)
            y.append(label)

        except:
            print("Error loading:", img_path)

X = np.array(X)
y = np.array(y)

print("Total Samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

joblib.dump(model, "pest_model.pkl")

print("Binary Pest Model saved as pest_model.pkl")
