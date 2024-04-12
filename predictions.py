import cv2
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image

from keras.applications.vgg16 import (
    decode_predictions,
    preprocess_input,
    VGG16,
)

model1 = VGG16(weights='imagenet')

# advers = load_img("advers_ct_ca_full_mut_noLanczos.png", target_size=(224,224))#, interpolation='lanczos')
advers = cv2.imread("advers_ct_HR_0411_eps8_10000.png")
advers = cv2.cvtColor(advers, cv2.COLOR_BGR2RGB)
advers = cv2.resize(advers, (224, 224))#, interpolation=cv2.INTER_LANCZOS4)

advers= img_to_array(advers)
image = advers.reshape((1, 224, 224, 3))
yhat = model1.predict(preprocess_input(image))
label0 = decode_predictions(yhat,1000)
label1 = label0[0][0]
print("The image isssss: %s  %.4f%%" % (label1[1], label1[2]))
print(label0)
print(label1[1], label1[2])