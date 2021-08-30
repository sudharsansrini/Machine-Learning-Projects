import cv2
import os

haar_file = 'haarcascade_frontalface_default.xml'

dataset = 'Datasets'
name = input('Enter name :')
path = os.path.join(dataset, name)
# print(path)
if not os.path.isdir(path):
    os.mkdir(path)
width , height = 200, 200

face_cascade = cv2.CascadeClassifier(haar_file)
cap = cv2.VideoCapture(0)
count = 1

while count < 31:
    print(count)
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = frame[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (path, count), face_resize)
        count += 1
    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()