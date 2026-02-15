import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================================
# DATASET PATH
# =====================================

dataset_path = "soil_dataset"

classes = ["dry_soil", "moist_soil", "waterlogged_soil"]

X = []
y = []

# =====================================
# LOAD IMAGES
# =====================================

for label, category in enumerate(classes):

    folder = os.path.join(dataset_path, category)

    if not os.path.exists(folder):
        print("Missing folder:", folder)
        continue

    for img_name in os.listdir(folder):

        img_path = os.path.join(folder, img_name)

        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128,128))

            # Convert to HSV (important for soil color detection)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            features = hsv.flatten()

            X.append(features)
            y.append(label)

        except:
            print("Error loading:", img_path)

# =====================================
# CONVERT TO NUMPY
# =====================================

X = np.array(X)
y = np.array(y)

print("Total Samples:", len(X))

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# TRAIN MODEL
# =====================================

model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# =====================================
# EVALUATE
# =====================================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# =====================================
# SAVE MODEL
# =====================================

joblib.dump(model, "soil_model.pkl")

print("Model saved as soil_model.pkl")
