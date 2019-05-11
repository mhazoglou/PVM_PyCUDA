##############
## Script listens to serial port and writes contents into a file
##############
## requires pySerial to be installed
import serial
from datetime import datetime
import cv2 as cv
import os
import time
import math as mt
import random


def reflecting_off_walls(start, stop, diff):
    r = range(start, stop, diff)
    L = len(r)
    def f(i):
        return r[i % L if (i // L % 2) == 0 else -1 - i % L]
    return f

def periodic_boundaries(start, stop, diff):
    r = range(start, stop, diff)
    L = len(r)
    def f(i):
        return r[i % L]
    return f


def oscillation(start, stop, diff, phase):
    r = range(start, stop, diff)
    L = len(r)
    def f(i):
        return int((stop - start) * mt.sin(
            mt.tau * (r[i % L] - start) / (stop - start) + phase
        ) / 2 + (start + stop) / 2)
    return f

def pan_random_trajectory():
    it = random.randint(0, 19)
    if it == 0:
        return reflecting_off_walls(0, 1024, 5)
    elif it == 1:
        return reflecting_off_walls(0, 1024, 10)
    elif it == 2:
        return reflecting_off_walls(0, 1024, 15)
    elif it == 3:
        return reflecting_off_walls(0, 1024, 20)
    elif it == 4:
        return reflecting_off_walls(1024, 0, -5)
    elif it == 5:
        return reflecting_off_walls(1024, 0, -10)
    elif it == 6:
        return reflecting_off_walls(1024, 0, -15)
    elif it == 7:
        return reflecting_off_walls(1024, 0, -20)
    elif it == 8:
        return periodic_boundaries(0, 1024, 5)
    elif it == 9:
        return periodic_boundaries(0, 1024, 10)
    elif it == 10:
        return periodic_boundaries(0, 1024, 15)
    elif it == 11:
        return periodic_boundaries(0, 1024, 20)
    elif it == 12:
        return periodic_boundaries(1024, 0, -5)
    elif it == 13:
        return periodic_boundaries(1024, 0, -10)
    elif it == 14:
        return periodic_boundaries(1024, 0, -15)
    elif it == 15:
        return periodic_boundaries(1024, 0, -20)
    elif it == 16:
        return oscillation(0, 1024, 5, mt.tau * random.random())
    elif it == 17:
        return oscillation(0, 1024, 10, mt.tau * random.random())
    elif it == 18:
        return oscillation(0, 1024, 15, mt.tau * random.random())
    elif it == 19:
        return oscillation(0, 1024, 20, mt.tau * random.random())
    elif it == 16:
        return oscillation(1024, 0, -5, mt.tau * random.random())
    elif it == 17:
        return oscillation(1024, 0, -10, mt.tau * random.random())
    elif it == 18:
        return oscillation(1024, 0, -15, mt.tau * random.random())
    elif it == 19:
        return oscillation(1024, 0, -20, mt.tau * random.random())

def tilt_random_trajectory():
    it = random.randint(0, 19)
    if it == 0:
        return reflecting_off_walls(256, 769, 5)
    elif it == 1:
        return reflecting_off_walls(256, 769, 10)
    elif it == 2:
        return reflecting_off_walls(256, 769, 15)
    elif it == 3:
        return reflecting_off_walls(256, 769, 20)
    elif it == 4:
        return reflecting_off_walls(769, 256, -5)
    elif it == 5:
        return reflecting_off_walls(769, 256, -10)
    elif it == 6:
        return reflecting_off_walls(769, 256, -15)
    elif it == 7:
        return reflecting_off_walls(769, 256, -20)
    elif it == 8:
        return periodic_boundaries(256, 769, 5)
    elif it == 9:
        return periodic_boundaries(256, 769, 10)
    elif it == 10:
        return periodic_boundaries(256, 769, 15)
    elif it == 11:
        return periodic_boundaries(256, 769, 20)
    elif it == 12:
        return periodic_boundaries(769, 256, -5)
    elif it == 13:
        return periodic_boundaries(769, 256, -10)
    elif it == 14:
        return periodic_boundaries(769, 256, -15)
    elif it == 15:
        return periodic_boundaries(769, 256, -20)
    elif it == 16:
        return oscillation(256, 769, 5, mt.tau * random.random())
    elif it == 17:
        return oscillation(256, 769, 10, mt.tau * random.random())
    elif it == 18:
        return oscillation(256, 769, 15, mt.tau * random.random())
    elif it == 19:
        return oscillation(256, 769, 20, mt.tau * random.random())
    elif it == 16:
        return oscillation(769, 256, -5, mt.tau * random.random())
    elif it == 17:
        return oscillation(769, 256, -10, mt.tau * random.random())
    elif it == 18:
        return oscillation(769, 256, -15, mt.tau * random.random())
    elif it == 19:
        return oscillation(769, 256, -20, mt.tau * random.random())

cap = cv.VideoCapture(-1)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FPS, 120)
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
n_frames = 1000
serial_port = '/dev/ttyUSB0'
baud_rate = 115200 #In arduino, Serial.begin(baud_rate)
path = "/media/sdb/MotionAndVisionIntegrationData/"

# What better place than here?
# What better time than now?
right_now = datetime.now()

timestamp = right_now.isoformat()
write_to_path = path + timestamp
write_to_img_path = write_to_path + "/img/"
write_to_file_path = write_to_path + "/" + "output.txt"

input_pan = pan_random_trajectory()
input_tilt = tilt_random_trajectory()
frame_list = []
out_list = []

# L_pan = len(input_pan)
# L_tilt = len(input_tilt)

with serial.Serial(serial_port, baud_rate) as ser:
    print("Sleeping for one second")
    time.sleep(1)
    
    print("writing on serial")
    ser.write((str(input_pan(0)) + "," 
               + str(input_tilt(0)) + "\n").encode())
    
    print("Sent")
    line1 = ser.readline()
    line1 = line1.decode("utf-8")
    print("Received. Should be working fine.")
    ret, frame = cap.read()
    
    print("Sleeping for one second")
    time.sleep(1)
    
    print("Start")
    
    for i in range(n_frames):
        pan = input_pan(i) #input_pan[i % L_pan if (i // L_pan % 2) == 0 else -1 - i % L_pan]
        tilt = input_tilt(i) #input_tilt[i % L_tilt if (i // L_tilt % 2) == 0 else -1 - i % L_tilt]
        send_string = str(pan) + ',' + str(tilt) + '\n'
        # ping controller for position
        ser.write(send_string.encode())
        line1 = ser.readline()
        line1 = line1.decode("utf-8")
        line_time_stamp1 = datetime.now()
        
        # capture frame
        ret, frame = cap.read()
        frame_time_stamp = datetime.now()
        
        print(line1[:-2] + ' at ' + line_time_stamp1.isoformat(' '))
        print('Frame captured at ' + frame_time_stamp.isoformat(' '))
        out_list.append(line_time_stamp1.isoformat('-') + ', ' + line1
                       )
        frame_list.append((write_to_img_path +
                           frame_time_stamp.isoformat("-") + '.png',
                           frame)
                         )

dir_list = os.listdir(path)

if timestamp not in dir_list:
    os.mkdir(write_to_path)
    os.mkdir(write_to_img_path)
else:
    data_list = os.listdir(write_to_path)
    if "img" not in data_list:
        os.mkdir(write_to_img_path)

with open(write_to_file_path, "w+") as output_file:
    for ((fname, frame), line)  in zip(frame_list, out_list):
        cv.imwrite(fname, frame)
        output_file.write(line)

cap.release()

