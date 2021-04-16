import cv2 as cv
 
cap = cv.VideoCapture("Face Rec/test.mp4")
face_cascade = cv.CascadeClassifier('Face Rec/haarcascade_frontalface_default.xml')
while cap.isOpened():
    _, frame = cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=3)

            
    cv.imshow("Face rec :", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
