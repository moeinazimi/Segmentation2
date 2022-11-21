import cv2
import numpy as np
 
# Read image
image = cv2.imread("results.png")
 
# Select ROI
r = cv2.selectROI("select the area", image)

Top_Left_X, Top_Left_Y, Width, Height = r

with open('roi.txt', 'w') as f:
    f.write(str(r).replace(')','').replace('(','').replace('.0',''))
    

# Select ROI
r = cv2.selectROI("select the area", image)

Top_Left_X, Top_Left_Y, Width, Height = r

with open('roi2.txt', 'w') as f:
    f.write(str(r).replace(')','').replace('(','').replace('.0',''))
    
    

# Select ROI
r = cv2.selectROI("select the area", image)

Top_Left_X, Top_Left_Y, Width, Height = r

with open('roi3.txt', 'w') as f:
    f.write(str(r).replace(')','').replace('(','').replace('.0',''))