import cv2
import numpy as np

input_file = 'AgandBio/2.jpg'
im = cv2.imread(input_file)

im_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gg = cv2.GaussianBlur(im_g, (5,5), 0)
clahe= cv2.createCLAHE(10, (21,21))
im_gahe = clahe.apply(im_gg)
cv2.imshow('gray', im_gg)
cv2.imshow('gray clahe', im_gahe)
_ , bw = cv2.threshold(im_gahe, 130, 255, cv2.THRESH_BINARY)
cv2.imshow('bw', bw)
black = np.zeros_like(im_gg)

bwed = cv2.dilate(cv2.erode(bw, np.ones((5,1))), np.ones((5,1)))

cv2.imshow('Horizontal Erased', bwed)

contours, _ = cv2.findContours(bwed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on size and aspect ratio
seedling_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)

    if area < 1250 and (aspect_ratio < 0.4 or (aspect_ratio < 1.5 and aspect_ratio > 0.5)):
        seedling_contours.append(contour)

mask = np.zeros_like(im_g)
cv2.drawContours(mask, seedling_contours, -1, (255), thickness=cv2.FILLED)

seedlings = cv2.bitwise_and(im, im, mask=mask)

black_background = np.zeros_like(im)

# Paste the extracted seedlings onto the black canvas
black_background[mask == 255] = seedlings[mask == 255]

cv2.imshow('final', black_background)
cv2.imwrite(f'{input_file[:-4]}_processed.jpg', black_background)

cv2.waitKey(0)
cv2.destroyAllWindows()