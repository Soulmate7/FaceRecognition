#coding=utf-8

import cv2
import dlib

def detect(filename):
    img=cv2.imread(filename)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #人脸分类器
    detector=dlib.get_frontal_face_detector()
    #获取人脸检测器
    predictor=dlib.shape_predictor(r"/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/shape_predictor_68_face_landmarks.dat")
    #dets保存的是图像中人脸的矩形框，可以有多个
    dets=detector(gray,1)
    
    for face in dets:
        shape=predictor(img,face)# 寻找人脸的68个标定点
        # 遍历所有点，打印出其坐标，并圈出来
        for pt in shape.parts():
            pt_pos=(pt.x,pt.y)
            cv2.circle(img,pt_pos,2,(0,255,0),1)
        cv2.imshow("image",img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    detect('image/101.png')