# Author: Hadeer Arafa
# Data :  10:32 pm 12/3/2021
blink_num=0
import cv2 as cv

face_cascade = cv.CascadeClassifier(cv.samples.findFile(cv.data.haarcascades + 'haarcascade_frontalface_default.xml'))
eye_cascade  = cv.CascadeClassifier(cv.samples.findFile(cv.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'))

first_read = True
capture = cv.VideoCapture(0)
isTrue, img = capture.read()
print('hi')
while (isTrue):
    isTrue, img = capture.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 5, 1, 1)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
    if (len(faces) > 0):
        for (x, y, w, h) in faces:
            img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 225, 0), 2)
            roi_face = gray[y:y + h, x:x + w]
            roi_face_clr = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))
            if (len(eyes) >= 2):
                if (first_read):
                    cv.putText(img,
                               "Eye detected press s to begin",
                               (70, 70),
                               cv.FONT_HERSHEY_PLAIN, 3,
                               (0, 255, 0), 2)
                else:
                    cv.putText(img,
                               "Eyes open !", (70, 70),
                               cv.FONT_HERSHEY_PLAIN, 2,
                               (255, 255, 255), 2)
            else:
                if (first_read):
                    cv.putText(img,
                               "No Eyes detected !", (70, 70),
                               cv.FONT_HERSHEY_PLAIN, 3,
                               (0, 0, 255), 2)
                else:
                    cv.putText(img,
                               " Eyes blink !", (70, 70),
                               cv.FONT_HERSHEY_PLAIN, 3,
                               (0, 0, 255), 2)
                    #blink_num=blink_num+1
                    #print("blink detected "+blink_num+" times")
                    #cv.waitKey(3000)
                    #first_read = True
    else:
        cv.putText(img,
                   "No Face detected !", (100, 100),
                   cv.FONT_HERSHEY_PLAIN, 3,
                   (0, 0, 255), 2)
    cv.imshow('img', img)
    a = cv.waitKey(1)
    if (a == ord('d')):
        break
    elif (a == ord('s') and first_read):
        first_read = False

capture.release()
cv.destroyAllWindows()


