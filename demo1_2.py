
import face_recognition
import cv2

image = face_recognition.load_image_file("/Users/zengrui/Desktop/媒体大数据/Code/FaceRecognition/face.png/Users/zengrui/Desktop/媒体大数据/Code/FaceRecognition/face.png")
face_locations_noCNN=face_recognition.face_locations(image)
# A list of tuples of found face locations in css (top, right, bottom, left) order
# 因为返回值的顺序是这样子的，因此在后面的for循环里面赋值要注意按这个顺序来



print("face_location_noCNN:")
print(face_locations_noCNN)
face_num2=len(face_locations_noCNN)
print(face_num2)       # The number of faces
# 到这里为止，可以观察两种情况的坐标和人脸数，一般来说，坐标会不一样，但是检测出来的人脸数应该是一样的
# 也就是说face_num1　＝　face_num２；　face_locations_useCNN　和　face_locations_noCNN　不一样


org = cv2.imread("/Users/zengrui/Desktop/媒体大数据/Code/FaceRecognition/face.png")
img = cv2.imread("/Users/zengrui/Desktop/媒体大数据/Code/FaceRecognition/face.png")
cv2.imshow("/Users/zengrui/Desktop/媒体大数据/Code/FaceRecognition/face.png",img)  # 原始图片




for i in range(0,face_num2):
    top = face_locations_noCNN[i][0]
    right = face_locations_noCNN[i][1]
    bottom = face_locations_noCNN[i][2]
    left = face_locations_noCNN[i][3]

    start = (left, top)
    end = (right, bottom)

    color = (0,255,255)
    thickness = 2
    cv2.rectangle(org, start, end, color, thickness)

cv2.imshow("no cnn ",org)

cv2.waitKey(0)
cv2.destroyAllWindows()



# # use CNN
# face_locations_useCNN = face_recognition.face_locations(image,model='cnn')
# model – Which face detection model to use. “hog” is less accurate but faster on CPUs.
# “cnn” is a more accurate deep-learning model which is GPU/CUDA accelerated (if available). The default is “hog”.

# print("face_location_useCNN:")
# print(face_locations_useCNN)
# face_num1=len(face_locations_useCNN)
# print(face_num1)       # The number of faces


# for i in range(0,face_num1):
#     top = face_locations_useCNN[i][0]
#     right = face_locations_useCNN[i][1]
#     bottom = face_locations_useCNN[i][2]
#     left = face_locations_useCNN[i][3]
#
#     start = (left, top)
#     end = (right, bottom)
#
#     color = (0,255,255)
#     thickness = 2
#     cv2.rectangle(img, start, end, color, thickness)    # opencv 里面画矩形的函数

# # Show the result
# cv2.imshow("useCNN",img)