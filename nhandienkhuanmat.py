import cv2
import face_recognition
import os
import numpy as np
# b1 load anh tu kho anh pic 2
path="pic2"
images = []
className = []
mylist =os.listdir(path)
print(mylist)
for cl in mylist:
    curimg = cv2.imread(f"{path}/{cl}")
    images.append(curimg)
    className.append(os.path.splitext(cl)[0])
print(len(images))
print(className)
# b2 ma hoa cac anh
def Mahoa(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeListKnow = Mahoa(images)
print("ma hoa thanh cong")
print(len(encodeListKnow))
# khoi dong webcam
cap= cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()
    frameS = cv2.resize(frame,(0,0),None,fx=0.5,fy=0.5)
    frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)

    # xac dinh vi tri
    facecurFrame = face_recognition.face_locations(frameS) # lay tung khuan mat vao vi tri khuan mat hien tai
    encodecurFrame = face_recognition.face_encodings(frameS)
    for encodeFace, faceLoc in zip(encodecurFrame,facecurFrame):
        matsches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis) # day ve gia tri index nho nhat

        if faceDis[matchIndex] <0.50 :
            name = className[matchIndex].upper()
        else:
            name = " nguoi la"
        # ve ten len anh
        y1, x2,y2,x1= faceLoc
        y1, x2, y2, x1= y1*2, x2*2,y2*2,x1*2
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,name,(x2,y2),cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2)
    cv2.imshow('cam quan sat', frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()