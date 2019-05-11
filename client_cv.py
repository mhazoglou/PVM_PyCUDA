import cv2 as cv
import numpy as np
import socket
import pickle
import struct
import serial

scale = 5
wait_time = 77
size = (128, 96)
# enter the IP address of the machine running the server script
IP_ADDRESS = '127.0.0.1'
PORT = 8089
buffer = 4096
serial_port, baud_rate = '/dev/ttyUSB0', 9600

cap = cv.VideoCapture(0)
# adjust resolution to 240p 
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
# set frame rate to 120 fps
cap.set(cv.CAP_PROP_FPS, 120)
# set buffer size to one frame to get the lastest frame
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

payload_size = struct.calcsize("L")

def rescale(img, scale=1):
    return cv.resize(img, (0, 0), fx=scale, fy=scale)


def unpack_received_data(data, socket):
    while len(data) < payload_size:
        data += socket.recv(buffer)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]

    # unpacking message size
    msg_size = struct.unpack("L", packed_msg_size)[0]

    # receiving data until complete
    while len(data) < msg_size:
        data += socket.recv(buffer)
    object_data = data[:msg_size]
    data = "".encode()

    # deserializing pickled data
    object = pickle.loads(object_data)
    return object, data


with serial.Serial(serial_port, baud_rate) as ser:
    try:
        with socket.socket(socket.AF_INET,
                           socket.SOCK_STREAM) as clientsocket:
            clientsocket.connect((IP_ADDRESS, PORT))

            rec_data = "".encode()

            while True:
                ret, frame = cap.read()
                frame = cv.resize(frame, size)
                data = pickle.dumps(frame)
                clientsocket.sendall(struct.pack("L", len(data)) + data)

                cv.namedWindow('Input frame',
                               cv.WINDOW_GUI_NORMAL | cv.WINDOW_FREERATIO)
                cv.namedWindow('Prediction',
                               cv.WINDOW_GUI_NORMAL | cv.WINDOW_FREERATIO)
                cv.namedWindow('Error in Prediction',
                               cv.WINDOW_GUI_NORMAL | cv.WINDOW_FREERATIO)

                cv.imshow('Input frame', rescale(frame, scale=scale))

                # receive a simple serial message for pan-tilt position,
                # error and prediction
                pack, rec_data = unpack_received_data(rec_data,
                                                      clientsocket)

                pan_tilt, pred_frame, err_frame = pack

                print('Received: "{}"'.format(pan_tilt))
                ser.write(pan_tilt)

                # showing the frames locally
                cv.imshow('Prediction', rescale(pred_frame, scale=scale))
                cv.imshow('Error in Prediction',
                          rescale(err_frame, scale=scale))
                cv.waitKey(wait_time)

    except KeyboardInterrupt:
        print('Recentering the position')
        ser.write('512,512,'.encode())
    
