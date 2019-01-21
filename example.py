import os
import cv2
import numpy as np

from pca_classifier import PCAClassifier

IMAGES_FOLDER = 'lfw1000'


def fix(x):
    x = int(x)
    if x < 0:
        return 0
    if x > 255:
        return 255
    return x


def main():
    images = [cv2.imread(os.path.join(IMAGES_FOLDER, path), cv2.IMREAD_GRAYSCALE)
              for path
              in os.listdir(IMAGES_FOLDER)]

    shape = images[0].shape

    test_image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
    test_image = cv2.resize(test_image, (shape[1], shape[0]))

    clf = PCAClassifier()
    clf.fit(images)

    # cv2.imshow('img', clf.mean.reshape(shape).astype(np.uint8))
    # cv2.waitKey(0)

    print(clf.predict(test_image, 80))


if __name__ == '__main__':
    main()

main()
