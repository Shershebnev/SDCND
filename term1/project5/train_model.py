import glob
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from constants import *
from helpers import extract_features

SEED = 42
np.random.seed(SEED)


def main():
    """Train and save the model and fitter scaler
    """
    cars = glob.glob("vehicles/*/*png")
    notcars = glob.glob("non-vehicles/*/*png")

    car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat,
                                    hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat,
                                       hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    X_scaler = StandardScaler().fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    svc = LinearSVC()
    svc.fit(X_train, y_train)
    score = svc.score(X_test, y_test)
    print("Finished training, score on test set: {:.2f}".format(score))

    pickle.dump(X_scaler, open("scaler.pkl", "wb"))
    pickle.dump(svc, open("svc.pkl", "wb"))

if __name__ == "__main__":
    main()