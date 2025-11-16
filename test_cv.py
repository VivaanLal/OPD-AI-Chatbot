import cv2

cap = cv2.VideoCapture(0)
print("Camera opened:", cap.isOpened())

while True:
    ret, frame = cap.read()
    print("Frame:", ret)
    if ret:
        cv2.imshow("Test", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
