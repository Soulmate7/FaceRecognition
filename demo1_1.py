import cv2

def detect(filename):
    face_cascade = cv2.CascadeClassifier('/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    #加载haar数据
    img=cv2.imread(filename)
    #加载图片,读进来直接是BGR格式
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
    #cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # 识别图像中的人脸，返回所有人脸的矩形框向量组
    # scaleFactor=1.3 为了检测到不同大小的目标，通过scalefactor参数把图像长宽同时按照一定比例1.3逐步缩小，
    # 然后检测，这个参数设置的越大，计算速度越快，但可能会错过了某个大小的人脸。
    # minNeighbors=5 构成检测目标的相邻矩形的最小个数，此处设置为5

    for(x,y,w,h) in faces:
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
    #在图像中画上矩形框
    cv2.imshow('Person Detected!',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #显示结果

if __name__ == '__main__':
    detect('image/101.png')