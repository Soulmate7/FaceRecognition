import cv2
import face_recognition
#显示已知图片

def match(img1,img2):
    known_image=cv2.imread("image/xl1.JPG")
    cv2.imshow("image", known_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    known_image = face_recognition.load_image_file("image/xl1.JPG")
    # 显示待匹配图片
    unknown_image=cv2.imread("image/xl2.JPG")
    cv2.imshow("unknown_image", unknown_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    unknown_image = face_recognition.load_image_file("image/xl2.JPG")

    known_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([known_encoding],
                                             unknown_encoding,
                                             tolerance=0.6)
    if results[0] == True:
        print("匹配成功，该未知图片与已有图片人脸可匹配！")
else:
    print("匹配失败！")

if __name__=='__main__':
    match("image/xl1.JPG","image/xl2.JPG")