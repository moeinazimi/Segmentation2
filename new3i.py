import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import joblib



image = skimage.io.imread('results.png')
mask = skimage.io.imread('mask.png')
training_labels = np.zeros(image.shape[:2], dtype=np.uint8)


indices = np.where(mask == [255])
print(len(indices[0]))
for i in range(len(indices[0])-300000):
    training_labels[indices[0][i], indices[1][i]] = 1
    
indices = np.where(mask == [0])
print(len(indices[0]))
for i in range(len(indices[0])):
    training_labels[indices[0][i], indices[1][i]] = 2

sigma_min = 1
sigma_max = 16
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=False, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max,
                        channel_axis=-1)
features = features_func(image)

print(features.shape)

clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                             max_depth=10, max_samples=0.05)
#joblib.dump(clf, "./random_forest.joblib")

clf = future.fit_segmenter(training_labels, features, clf)

print(clf)
joblib.dump(clf, "./random_forest.joblib")

result = future.predict_segmenter(features, clf)



fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
ax[0].imshow(segmentation.mark_boundaries(image, result, mode='thick'))
ax[0].contour(training_labels)
ax[0].set_title('Image, mask and segmentation boundaries')
ax[1].imshow(result)
ax[1].set_title('Segmentation')
fig.tight_layout()
plt.show()

'''
sigma_min = 1
sigma_max = 16
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=False, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max,
                        channel_axis=-1)
loaded_rf = joblib.load("./random_forest.joblib")
image = skimage.io.imread('1.jpg')
features = features_func(image)
result = future.predict_segmenter(features,loaded_rf)


fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
ax[0].imshow(segmentation.mark_boundaries(image, result, mode='thick'))
#ax[0].contour(training_labels)
ax[0].set_title('Image, mask and segmentation boundaries')
ax[1].imshow(result)
ax[1].set_title('Segmentation')
fig.tight_layout()
plt.show()
'''