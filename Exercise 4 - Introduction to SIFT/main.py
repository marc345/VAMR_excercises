import cv2
import numpy as np
import matplotlib.pyplot as plt

num_scales = 3  # Scales per octave.
num_octaves = 5  # Number of octaves.
sigma_0 = 1.6
contrast_threshold = 0.04
image_file_1 = 'images/img_1.jpg'
image_file_2 = 'images/img_2.jpg'
rescale_factor = 0.2;  # Rescaling of the original image for speed

img1 = cv2.imread(image_file_1, cv2.IMREAD_COLOR)
img2 = cv2.imread(image_file_2, cv2.IMREAD_COLOR)
img1 = cv2.resize(img1, (0,0), fx=2.0, fy=2.0)
img2 = cv2.resize(img2, (0,0), fx=2.0, fy=2.0)

imgs = [img1, img2]

# list holding the 3D DoG volume for the different octaves
dog_list = []

for img in imgs:
    # Write code to compute:
    # 1)    image pyramid. Number of images in the pyramid equals
    #       'num_octaves'.
    # 2)    blurred images for each octave. Each octave contains
    #       'num_scales + 3' blurred images.
    # 3)    'num_scales + 2' difference of Gaussians for each octave.
    for o in range(num_octaves):
        if o != 0:
            sampling_factor = 2 ** -o
            img = cv2.resize(img1, (0,0), fx=2.0, fy=2.0)
        # generate blurred versions of the input images with Gaussian filters with differnt sigma
        blurrs = []
        for s in range(-1, num_scales+1, 1):
            sigma = sigma_0 * (2 ** (s/num_scales))
            # 99% of the signal are contained within the [-3*sigma, 3*sigma] interval
            # chose kernel size accordingly, i.e. that Gaussian decays to zero within kernel and
            # does not sharply drop off (would introduce high frequency noise)
            kernel_size = int(2 * np.ceil(3*sigma) + 1)
            blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
            blurrs.append(blurred)

        # compute difference of Gaussians (DoGs)
        dogs = []
        for b in range(1, len(blurrs)):
            dog = blurrs[b] - blurrs[b-1]
            dogs.append(dog)

        dog_list.append(dogs)


# 4)    Compute the keypoints with non-maximum suppression and
#       discard candidates with the contrast threshold.
# 5)    Given the blurred images and keypoints, compute the
#       descriptors. Discard keypoints/descriptors that are too close
#       to the boundary of the image. Hence, you will most likely
#       lose some keypoints that you have computed earlier.

# Finally, match the descriptors using the function 'matchFeatures' and
# visualize the matches with the function 'showMatchedFeatures'.
# If you want, you can also implement the matching procedure yourself using
# 'knnsearch'.