from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import numpy as np
# import matplotlib.pyplot as plt
import time

car_features = np.load(file='../saved_features/rgb_all/car_features.npy')
non_car_features = np.load(file='../saved_features/rgb_all/non_car_features.npy')

print(car_features.shape)
print(non_car_features.shape)


x = np.vstack((car_features, non_car_features)).astype(np.float64)
# Fit a per-column scaler
x_scaler = StandardScaler().fit(x)
# Apply the scaler to X
scaled_x = x_scaler.transform(x)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
x_train, x_test, y_train, y_test = train_test_split(scaled_x, y,
                                                    test_size=0.2, random_state=rand_state)

print('Feature vector length:', len(x_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(x_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(x_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(x_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

joblib.dump(svc, '../saved_models/svc_rgb_all.pkl')
joblib.dump(x_scaler, '../saved_models/x_scaler_rgb_all.pkl')

