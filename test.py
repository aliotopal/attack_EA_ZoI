import numpy as np
import random

roi = np.load("unique_pixels.npy")

print(roi.shape)
img = (500, 374)

ratio = 100 * roi.shape[0] / (img[0]*img[1])
print("The percentage of the image modified is: %.1f " %(ratio))


no_of_pixels = random.randrange(0, len(roi))
pixel_idx = random.sample(range(1, len(roi)), no_of_pixels)

print("No of pixels: ", no_of_pixels)
print("pizel_idx: ", pixel_idx)