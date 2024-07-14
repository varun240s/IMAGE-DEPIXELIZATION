import cv2
import numpy as np

# read original and pixelated image
img1 = cv2.imread(r"C:\Users\reddy\Desktop\INTEL UNNATI\clear.png")
img2 = cv2.imread(r"C:\Users\reddy\Desktop\INTEL UNNATI\blured.png")

# convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# compute laplacians
laplacian1 = cv2.Laplacian(gray1,cv2.CV_64F)
laplacian2 = cv2.Laplacian(gray2,cv2.CV_64F)

# get variances
variance1 = np.var(laplacian1)
variance2 = np.var(laplacian2)
print ('variance of original image:', variance1)
print ('variance of pixelated image:', variance2)

# save images
# cv2.imwrite('mandril3_laplacian.png', (255*laplacian1).clip(0,255).astype(np.uint8))
# cv2.imwrite('mandril3_pixelated_laplacian.png', (255*laplacian2).clip(0,255).astype(np.uint8))

# show laplacian using OpenCV
cv2.imshow("laplacian1", laplacian1)
cv2.imshow("laplacian2", laplacian2)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
variance of original image: 986.0862773427732
variance of pixelated image: 576.2855625

"""