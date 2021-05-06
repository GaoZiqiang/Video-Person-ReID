from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
yolo = YOLO()


capture=cv2.VideoCapture("video/2.mp4")

frame_interval = 6  #每frame_interval帧就做一次检测
frame_now = 0  #记录当前帧数
while(True):
    # 读取某一帧
    ref,frame=capture.read()
    frame_now += 1

    if(frame_now % frame_interval == 0):
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))

        # 进行检测
        frame = np.array(yolo.detect_image(frame))

        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("video", frame)

        c = cv2.waitKey(1) & 0xff
        if c == 27:
            capture.release()
            break

