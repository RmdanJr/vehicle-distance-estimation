# display results image
# imread() reads image as grayscale, second argument is one => grayscale, zero => RGB 
img = cv2.imread('/content/yolov5/runs/train/exp/results.png', 0)
plt.imshow(img)
plt.axis('off')
