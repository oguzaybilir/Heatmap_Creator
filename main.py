import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
import heatmap_creator


model_builder = DenseNet121
img_size = (224,224)

last_conv_layer_name = 'conv5_block16_2_conv'

img_path = "/home/oguzay/Documents/GitHub/Heatmap_Creator/Heatmap_Creator/src/no/68.png"

image_ = cv2.imread(img_path)

img_array = preprocess_input(heatmap_creator.get_img_array(img_path, size=img_size))

modelH = model_builder(weights='imagenet')

modelH.layers[-1].activation = None

preds = modelH.predict(img_array)
print("Predicted  : ", decode_predictions(preds, top=1)[0])

heatmap = heatmap_creator.make_gradcam_heatmap(img_array, modelH, last_conv_layer_name)
path_to_save_up = "/home/oguzay/Documents/GitHub/Heatmap_Creator/Heatmap_Creator/outputs/output.jpg"
heatmap_out = heatmap_creator.save_and_display_gradcam(img_path, heatmap, cam_path=path_to_save_up, alpha=0.5)


plt.figure()
plt.imshow(heatmap)
plt.show()

plt.figure()
plt.imshow(heatmap_out)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()