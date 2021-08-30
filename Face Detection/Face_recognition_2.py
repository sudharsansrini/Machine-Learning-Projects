import cv2
import os
import numpy as np

haar_file ='haarcascade_frontalface_default.xml'
dataset ='Datasets'
print('Training')
face_cascade = cv2.CascadeClassifier(haar_file)
images, labels, names, id = [], [], {}, 0
model = cv2.face.LBPHFaceRecognizer_create()

for subdirs, dirs, files in os.walk(dataset):
    # print(dirs)
    for subdir in dirs:
        print(subdir)
        names[id] = subdir
        subject_path = os.path.join(dataset, subdir)
        print(os.listdir(subject_path))
        for filename in os.listdir(subject_path):
            path = subject_path+'/'+filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id+=1
images, labels = [np.array(lis) for lis in [images, labels]]
print(images, labels)
width, height = 200, 200

model.train(images, labels)

cap = cv2.VideoCapture(0)
count = 1
while True:
    # print(count)
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        prediction = model.predict(face_resize)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)

        if prediction[1] < 800:
            cv2.putText(frame, '%s-%.0f'% (names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            print(names[prediction[0]])
            count=0
        else:
            count += 1
            cv2.putText(frame, 'Unknown', (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            if(count > 100):
                print("Unknown Person")
                count=0

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

