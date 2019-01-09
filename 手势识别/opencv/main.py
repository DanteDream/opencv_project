import cv2
import numpy as np
from PIL import Image
import math
from matplotlib import pyplot as plt
from threading import Event


c=1
writeImage=0

# 移除视频数据的背景噪声
def _remove_background(frame):
    fgbg = cv2.createBackgroundSubtractorMOG2() # 利用BackgroundSubtractorMOG2算法消除背景
    # fgmask = bgModel.apply(frame)
    fgmask = fgbg.apply(frame)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("do front",fgmask)
    kernel = np.ones((3, 3), np.uint8)
    # fgmask = cv2.erode(fgmask, kernel, iterations=1)

    fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel,iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res
# 视频数据的人体皮肤检测
def _bodyskin_detetc(frame):
    # 肤色检测: YCrCb之Cr分量 + OTSU二值化
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb) # 分解为YUV图像,得到CR分量
    (_, cr, _) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0) # 高斯滤波
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # OTSU图像二值化
    return skin

def _get_contours(array):
        # 利用findContours检测图像中的轮廓, 其中返回值contours包含了图像中所有轮廓的坐标点
     _, contours, _ = cv2.findContours(array, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
     return contours



def _get_eucledian_distance(a,b):
    distance=math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
    return distance



def _get_defects_count(array, contour, defects):
    ndefects = 0
    for i in range(defects.shape[0]):
        s,e,f,_ = defects[i,0]
        beg= tuple(contour[s][0])
        end= tuple(contour[e][0])
        far= tuple(contour[f][0])

        a= _get_eucledian_distance(beg, end)
        b= _get_eucledian_distance(beg, far)
        c= _get_eucledian_distance(end, far)

        angle   = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) # * 57


        if angle <= math.pi/2 :#90:
            # print("angle:\n", angle)
            ndefects = ndefects + 1
            cv2.circle(array, far, 3, (255,0,0), -1)
        cv2.line(array, beg, end,(255,0,0), 1)

        # cv2.imshow("看一看哦",array)

    return array, ndefects+1


def getPicShadow(frame,lag):

    print(frame[1][0])


# def grdetect(event,x,y,flags,param):
#     if event==cv2.EVENT_LBUTTONDOWN:
#         array=frame.copy()
#         copy= array.copy()
#
#
#
#
#         #灰度直方图
#         # gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
#         # cv2.imshow("灰度图", gray)
#         # min_val, max_val, min_indx, max_indx=cv2.minMaxLoc(gray)
#         # height=gray.shape[0]
#         # width=gray.shape[1]
#         # print("min",min_indx)
#         # plt.hist(gray.ravel(), 256, [0, 256])
#         # plt.show()
#
#         array = _remove_background(array) # 移除背景, add by wnavy
#         thresh = _bodyskin_detetc(array)
#         #去白噪点
#         # element=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
#         # thresh=cv2.dilate(thresh,element)
#
#         contours= _get_contours(thresh.copy()) # 计算图像的轮廓
#         largecont  = max(contours, key = lambda contour: cv2.contourArea(contour))
#
#         #想办法去掉阴影
#         # for i in range(height):
#         #     for j in range(width):
#         #         if([[i,j]] in largecont):
#         #             if(gray[i,j]<min_val+20):
#         #                 gray[i,j]=max_val
#         # gray2bgr=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
#         # cv2.imshow("abondunt",gray2bgr)
#
#         hull= cv2.convexHull(largecont) # 计算轮廓的凸点
#         hulls=cv2.convexHull(largecont,returnPoints = False)
#         defects= cv2.convexityDefects(largecont, hulls) # 计算轮廓的凹点
#         #print(len(contours))
#         num=len(defects)
#         # print(hull)
#         cv2.drawContours(copy, largecont, -1, (0, 255, 255), 10)
#         for i in range(num):
#             cv2.drawContours(copy,largecont,i,(0,255,255),10)#全部绘制设为-1
#             cv2.polylines(copy, [hull], True, (0, 255, 0), 2)
#
#         array, ndefects=_get_defects_count(copy, largecont, defects)
#
#
#
#         print("一共有",ndefects,"个手指头")
#
#         cv2.imshow("bg",array)
#         cv2.imshow("thresh",thresh)
#         cv2.imshow("a",copy)
#
#     else:
#         return 1

# def grdetect(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         array = frame.copy()
#         copy = array.copy()
#
#         array = _remove_background(array)  # 移除背景, add by wnavy
#         thresh = _bodyskin_detetc(array)
#         # 去白噪点
#         # element=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
#         # thresh=cv2.dilate(thresh,element)
#
#         contours = _get_contours(thresh.copy())  # 计算图像的轮廓
#         largecont = max(contours, key=lambda contour: cv2.contourArea(contour))
#
#         # 想办法去掉阴影
#         # for i in range(height):
#         #     for j in range(width):
#         #         if([[i,j]] in largecont):
#         #             if(gray[i,j]<min_val+20):
#         #                 gray[i,j]=max_val
#         # gray2bgr=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
#         # cv2.imshow("abondunt",gray2bgr)
#
#         hull = cv2.convexHull(largecont)  # 计算轮廓的凸点
#         hulls = cv2.convexHull(largecont, returnPoints=False)
#         defects = cv2.convexityDefects(largecont, hulls)  # 计算轮廓的凹点
#         # print(len(contours))
#         num = len(defects)
#         # print(hull)
#         cv2.drawContours(copy, largecont, -1, (0, 255, 255), 10)
#         for i in range(num):
#             cv2.drawContours(copy, largecont, i, (0, 255, 255), 10)  # 全部绘制设为-1
#             cv2.polylines(copy, [hull], True, (0, 255, 0), 2)
#
#         array, ndefects = _get_defects_count(copy, largecont, defects)
#
#         print("一共有", ndefects, "个手指头")
#
#         cv2.imshow("bg", array)
#         cv2.imshow("thresh", thresh)
#         cv2.imshow("a", copy)
#
#     else:
#         return 1

def grdetectOr(frame,c,writeImage):#静态图实验
    a = str(writeImage)
    array = frame.copy()
    copy = array.copy()

    array = _remove_background(array)  # 移除背景, add by wnavy
    thresh = _bodyskin_detetc(array)
    if (c % timeF == 0 & writeImage <= 10):
        cv2.imwrite("G:\\ImageTest\\GESTURE\\opencv_gesture_back\\"+a+".jpg",array)#存储去背景图片
        cv2.imwrite("G:\\ImageTest\\GESTURE\\opencv_gesture_ycrcb\\" + a + ".jpg", thresh)  # 存储肤色识别后的图片
    # 去白噪点
    # element=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # thresh=cv2.dilate(thresh,element)

    contours = _get_contours(thresh.copy())  # 计算图像的轮廓
    largecont = max(contours, key=lambda contour: cv2.contourArea(contour))

    # 想办法去掉阴影
    # for i in range(height):
    #     for j in range(width):
    #         if([[i,j]] in largecont):
    #             if(gray[i,j]<min_val+20):
    #                 gray[i,j]=max_val
    # gray2bgr=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    # cv2.imshow("abondunt",gray2bgr)

    hull = cv2.convexHull(largecont)  # 计算轮廓的凸点
    hulls = cv2.convexHull(largecont, returnPoints=False)
    defects = cv2.convexityDefects(largecont, hulls)  # 计算轮廓的凹点
    # print(len(contours))
    num = len(defects)
    # print(hull)
    cv2.drawContours(copy, largecont, -1, (0, 255, 255), 10)
    for i in range(num):
        cv2.drawContours(copy, largecont, i, (0, 255, 255), 10)  # 全部绘制设为-1
        cv2.polylines(copy, [hull], True, (0, 255, 0), 2)

    array, ndefects = _get_defects_count(copy, largecont, defects)

    print("一共有", ndefects, "个手指头")
    if (c % timeF == 0 & writeImage <= 10):
        cv2.imwrite("G:\\ImageTest\\GESTURE\\opencv_gesture\\"+a+".jpg",frame)
        cv2.imwrite("G:\\ImageTest\\GESTURE\\opencv_gesture_check\\" + a + ".jpg", copy)

    # cv2.imshow("bg", array)
    # cv2.imshow("thresh", thresh)
    # cv2.imshow("a", copy)


cap = cv2.VideoCapture(0)
win="capture"

timeF=100
while(1):
    ret, frame = cap.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # 对捕获到的图像进行双边滤波
    image = Image.fromarray(frame)  # 转换图像数据格式

    # cv2.setMouseCallback(win, grdetect)


    grdetectOr(frame, c, writeImage)
    if(c%timeF==0&writeImage<=10):
        b=str(writeImage)
        cv2.imwrite("G:\\ImageTest\\GESTURE\\opencv_gesture_img\\"+b+".jpg",frame)#存储原图
        writeImage+=1



    c+=1
    cv2.imshow(win, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# src=cv2.imread("G:/ImageTest/GESTURE/test/5.jpg")
# src = cv2.bilateralFilter(src, 5, 50, 100)  # 对捕获到的图像进行双边滤波
# image = Image.fromarray(src)  # 转换图像数据格式
# grdetectOr(src)

cap.release()

cv2.waitKey(0)
# cap.release()
cv2.destroyAllWindows()