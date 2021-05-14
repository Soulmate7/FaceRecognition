import cv2

def detect(filename):
    face_cascade = cv2.CascadeClassifier('/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    img = cv2.imread("/Users/zengrui/Pictures/图片/足球队定妆照/IMG_2269.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


    cv2.imshow('Person Detected!', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect('test7.jpg')
