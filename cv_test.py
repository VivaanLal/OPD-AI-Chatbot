import cv2

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
print("Camera opened:", cap.isOpened())

while True:
    ret, frame = cap.read()
    print("Frame:", ret)
    if ret:
        cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()