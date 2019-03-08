# -*- coding:utf-8 -*-

import socket
import threading
import struct
import os
import time
import sys
import numpy as np
import cv2
import random
from detection import Detection

BASE_DIR = os.getcwd()
SRC = os.path.join(BASE_DIR, 'src')
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_GRAPH = os.path.join(SRC, 'models', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(SRC, 'labels', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 1


class stream_connection:

    def __init__(self, host, resolution, img_quality=15, interval=0):
        self.host = host
        self.resolution = resolution
        self.quality = 911 + 15
        self.interval = interval
        self.img_quality = img_quality

        self._set_socket()
        self.mutex = threading.Lock()
        self.path = os.getcwd()

    def _set_socket(self):
        self.socket = socket.socket()
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.host)
        self.socket.listen(1000)
        print("Server running on port:%d" % self.host[1])

    def process(self, connection, detection):
        # 随机生成一个窗口名,仅用于测试，上线后不需要在窗口显示
        window = str(random.randint(0, 100))
        # 发送要接受的视频参数
        connection.send(struct.pack("lhh", self.quality, self.resolution[0], self.resolution[1]))
        try:
            self._subprocess(connection, window, detection)
        except:
            connection.close()

        # display_thread=threading.Thread(target=self._process)
        # display_thread.start()
        # if interval != 0:   # 非0则启动保存截图到本地的功能
        #     saveThread=threading.Thread(target=self._savePicToLocal,args = (interval,
        #         ))
        #     saveThread.setDaemon(1)
        #     saveThread.start()

    def _subprocess(self, connection, window, detection):
        print('getting into process')
        while True:
            # 接收到视频数据
            info = struct.unpack("lhh", connection.recv(8))
            buffer_size = info[0]
            # print(buffer_size)
            if buffer_size:
                try:
                    self.mutex.acquire()
                    # print('get buffer')
                    self.buf = b''
                    # cv2.namedWindow(window)
                    while buffer_size > 0:
                        print('buffer size: ', buffer_size)  # 循环读取到一张图片的长度
                        temp_buf = connection.recv(buffer_size)
                        # print('buffer received')
                        buffer_size -= len(temp_buf)
                        self.buf += temp_buf
                        # print('transform buffer')
                        data = np.frombuffer(self.buf, dtype=np.int8)
                        # print('decoding')
                        self.image = cv2.imdecode(data, 1)
                        prediction = detection.process(self.image)
                        image_avec_boxes = detection.draw_detection()
                        # print(self.image.shape)
                        cv2.imshow(window, image_avec_boxes)
                except:
                    print("接收失败")
                    pass
                finally:
                    self.mutex.release()
                    if cv2.waitKey(10) == 32:
                        connection.close()
                        cv2.destroyAllWindows()
                        print("停止接收")
                        break

    # def _savePicToLocal(self, interval):
    #  while(1):
    #      try:
    #          self.mutex.acquire()
    #          path=os.getcwd() + "\\" + "savePic"
    #          if not os.path.exists(path):
    #              os.mkdir(path)
    #          cv2.imwrite(path + "\\" + time.strftime("%Y%m%d-%H%M%S",
    #                  time.localtime(time.time())) + ".jpg",self.image)
    #      except:
    #          pass
    #      finally:
    #          self.mutex.release()
    #          time.sleep(interval)

    # def check_config(self):
    #     path=os.getcwd()
    #     print(path)
    #     if os.path.isfile(r'%s\video_config.txt'%path) is False:
    #         f = open("video_config.txt", 'w+')
    #         print("w = %d,h = %d" %(self.resolution[0],self.resolution[1]),file=f)
    #         #print("IP is %s:%d" %(self.remoteAddress[0],self.remoteAddress[1]),file=f)
    #         print("Save pic flag:%d" %(self.interval),file=f)
    #         print("image's quality is:%d,range(0~95)"%(self.img_quality),file=f)
    #         f.close()
    #         print("初始化配置")
    #     else:
    #         f = open("video_config.txt", 'r+')
    #         tmp_data=f.readline(50)#1 resolution
    #         num_list=re.findall(r"\d+",tmp_data)
    #         self.resolution[0]=int(num_list[0])
    #         self.resolution[1]=int(num_list[1])
    #         tmp_data=f.readline(50)#2 ip,port
    #         num_list=re.findall(r"\d+",tmp_data)
    #         str_tmp="%d.%d.%d.%d"\
    #         %(int(num_list[0]),int(num_list[1]),int(num_list[2]),int(num_list[3]))
    #         self.remoteAddress=(str_tmp,int(num_list[4]))
    #         tmp_data=f.readline(50)#3 savedata_flag
    #         self.interval=int(re.findall(r"\d",tmp_data)[0])
    #         tmp_data=f.readline(50)#3 savedata_flag
    #         #print(tmp_data)
    #         self.img_quality=int(re.findall(r"\d+",tmp_data)[0])
    #        #print(self.img_quality)
    #         self.src=911+self.img_quality
    #         f.close()
    #         print("读取配置")

    def run(self):

        detect = Detection(
            graph=PATH_TO_GRAPH,
            labels=PATH_TO_LABELS,
            classes=NUM_CLASSES
        )
        try:
            while True:
                connection, addr = self.socket.accept()
                client_thread = threading.Thread(
                    target=self.process,
                    args=(connection, detect)
                )  # 有新的连接建立时，创建新线程
                client_thread.start()
        except:
            detect.terminate()
            print('Exceptions Occurred !')


def main():
    cam = stream_connection(host=('192.168.1.58', 8888), resolution=(640, 480))
    cam.run()


if __name__ == "__main__":
    main()
