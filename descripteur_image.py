import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

# folder = '.\Flickr8k\queries'
# q = os.listdir(folder)

# tmp_descripteur = []
# tmp_name = []

# for filename in q[0:5]:
#     path = os.path.join(folder, filename).replace(os.sep, '/')
#     img = cv.imread(path)

#     red_histo = np.ndarray.flatten(
#         cv.calcHist([img], [0], None, [256], [0, 256]))
#     green_histo = np.ndarray.flatten(
#         cv.calcHist([img], [1], None, [256], [0, 256]))
#     blue_histo = np.ndarray.flatten(
#         cv.calcHist([img], [2], None, [256], [0, 256]))

#     cv.normalize(red_histo, red_histo)
#     cv.normalize(green_histo, green_histo)
#     cv.normalize(blue_histo, blue_histo)

#     img_normalize = np.concatenate((red_histo, green_histo, blue_histo))
#     tmp_descripteur.append(img_normalize)
#     tmp_name.append([filename])

# descripteur = np.array(tmp_descripteur)
# filename_image = np.array(tmp_name)

# np.save('./descripteur', descripteur)
# np.save('./filename_image', filename_image)

# cv.imshow('image',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# img = cv.imread('./Flickr8k/queries/1130401779_8c30182e3e.jpg')


def descripteur(src_path, dst):
    ld = os.listdir(src_path)

    tmp_descripteur = []
    tmp_name = []

    for filename in ld:
        path = os.path.join(src_path, filename).replace(os.sep, '/')
        img = cv.imread(path)

        red_histo = np.ndarray.flatten(
            cv.calcHist([img], [0], None, [256], [0, 256]))
        green_histo = np.ndarray.flatten(
            cv.calcHist([img], [1], None, [256], [0, 256]))
        blue_histo = np.ndarray.flatten(
            cv.calcHist([img], [2], None, [256], [0, 256]))

        cv.normalize(red_histo, red_histo)
        cv.normalize(green_histo, green_histo)
        cv.normalize(blue_histo, blue_histo)

        img_normalize = np.concatenate((red_histo, green_histo, blue_histo))
        tmp_descripteur.append(img_normalize)
        tmp_name.append([filename])

    descripteur = np.array(tmp_descripteur)
    filename_image = np.array(tmp_name)

    np.save(dst, descripteur)
    np.save(f'{dst}_filename', filename_image)


descripteur('./Flickr8k/images', './descripteur_img')
