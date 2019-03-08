import socket
import struct
import cv2
import time
import numpy


class webcam:

    def __init__(self, remote):
        self.resolution = [640, 480]
        self.remote = remote
        self.img_quality = 15

        self._set_socket()

    def _set_socket(self):
        self.socket = socket.socket()
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def connect(self):
        # 建立连接
        self._set_socket()
        print('connect')
        if self.socket.connect_ex(self.remote) == 0:
            print('Connected !')
            return 1
        else:
            return 0

    def recv_config(self):
        # 接收视频参数
        info = struct.unpack("lhh", self.socket.recv(8))
        if info[0] > 911:
            print('Video setting received', 'info:', info)
            self.img_quality = int(info[0]) - 911
            self.resolution[0] = info[1]
            self.resolution[1] = info[2]
            self.resolution = tuple(self.resolution)
            return 1
        else:
            print('Nothing received')
            return 0

    def streaming(self):
        if self.recv_config() == 0:
            return

        # 启动相机
        camera = cv2.VideoCapture(0)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.img_quality]
        with open("video_info.txt", 'a+') as f:
            print("Got connection from %s:%d" % (self.remote[0], self.remote[1]), file=f)
            print("像素为:%d * %d" % (self.resolution[0], self.resolution[1]), file=f)
            print("打开摄像头成功", file=f)
            print("连接开始时间:%s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), file=f)

        while True:
            time.sleep(0.13)
            grabbed, img = camera.read()
            img = cv2.resize(img, self.resolution)
            result, img = cv2.imencode('.jpg', img, encode_param)
            img = numpy.array(img)
            self.imgdata = img.tostring()
            try:
                # 向服务器发送图片数据大小，分辨率，数据
                self.socket.send(
                    struct.pack("lhh", len(self.imgdata), self.resolution[0], self.resolution[1]) + self.imgdata)
            except:
                with open("video_info.txt", 'a+') as f:
                    print("%s:%d disconnected!" % (self.remote[0], self.remote[1]), file=f)
                    print("连接结束时间:%s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), file=f)
                    print("****************************************", file=f)
                    camera.release()
                return


def main():
    print("Connecting...")
    cam = webcam(remote=("192.168.1.58", 8888))
    # cam.check_config()
    # print("像素为:%d * %d"%(cam.resolution[0],cam.resolution[1]))
    # print("目标ip为%s:%d"%(cam.remoteAddress[0],cam.remoteAddress[1]))
    if cam.connect():
        cam.streaming()


if __name__ == "__main__":
    main()
