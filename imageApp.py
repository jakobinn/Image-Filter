import socket
import cv2
import numpy
import numpy as np
import threading
from multiprocessing.pool import ThreadPool
import time

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process_threaded(img, filters, threadn = 8):
    accum = np.zeros_like(img)
    def f(kern):
        return cv2.filter2D(img, cv2.CV_8UC3, kern)
    pool = ThreadPool(processes=threadn)
    for fimg in pool.imap_unordered(f, filters):
        np.maximum(accum, fimg, accum)
    return accum



TCP_IP = "0.0.0.0"
TCP_PORT = 5001

Running_period = 1


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(True)

while True:
 try:
     start_time = time.time()
     conn, addr = s.accept()
     print addr
     #thread = threading.Thread(target=handle_the_conn, args=(conn, addr))
     #thread.start()

    length = recvall(conn, 16)
    stringData = recvall(conn, int(length))
    choice = recvall(conn, 16)
    print choice
    choice = choice.strip()

    data = numpy.fromstring(stringData, dtype='uint8')
    decimg = cv2.imdecode(data, 1)

    # Image Processing Part
    if (choice == '1'):
        filters = build_filters()
        res2 = process_threaded(decimg, filters)
    elif (choice == '2'):
        res2 = cv2.cvtColor(decimg, cv2.COLOR_BGR2GRAY)
    elif (choice == '3'):
        kernel = np.ones((5, 5), np.float32) / 25
        res2 = cv2.filter2D(decimg,-1,kernel)
    elif (choice == '4'):
        img = cv2.GaussianBlur(decimg, (3, 3), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ddepth = cv2.CV_64F
        ksize = 1
        scale = 1
        delta = 0
        gray_lap = cv2.Laplacian(gray, ddepth, ksize=ksize, scale=scale, delta=delta)
        res2 = cv2.convertScaleAbs(gray_lap)
    else:
        res2 = decimg

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, imgencode = cv2.imencode('.jpg', res2, encode_param)
    data = numpy.array(imgencode)
    stringData = data.tostring()

    print 'send'
    if Running_Server != 1:
        conn.send(("O" + str(len(stringData))).ljust(16))
    else:
        conn.send(str(len(stringData)).ljust(16))
    conn.send(stringData)

     Running_period -= 1

     #Check the threads:
     if Running_period <= 0:
         if time.time() - start_time < 2000:
            start_AWS()
            Running_Server += 1
            Reload(start_time, Running_period)
         else:
            Reload(start_time, Running_period)
     elif Running_period > 10 and Running_Server > 1:
         if time.time() - start_time > 10000:
            kill_AWS()
            Running_Server -= 1

     print start_time


 except s.error:
     break

s.close()

#cv2.imshow('SERVER', decimg)
#cv2.waitKey(0)
#cv2.destroyAllWindows()