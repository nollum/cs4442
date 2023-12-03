import cv2 
import matplotlib.pyplot as plt 

image = cv2.imread('salad.jpg') 

# Display the original image
plt.figure(1)
plt.title("Original Image") 
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

channels = cv2.split(image_hsv)
channel_names = ("Hue", "Saturation", "Value")

plt.figure(2)
plt.title("HSV Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []

for (channel, name) in zip(channels, channel_names):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    features.extend(hist.flatten())
    
    plt.plot(hist, label=name)

plt.legend()  
plt.xlim([0, 256])
plt.show()

