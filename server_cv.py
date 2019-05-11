import socket
import pickle
import numpy as np
import struct ## new
from pycuda import gpuarray
import os

from FormattingFiles import flatten_image, unflatten_image
# importing a function to give a connection dictionary
from RectangularGridConstructor import make_connections, break_stuff

# pick your device the default is 0 if not specified
# if the next line is not commented
os.environ['CUDA_DEVICE'] = '1' 

# autoinit automatically initializes a CUDA context
import pycuda.autoinit

from PVM_PyCUDA import PhantomXTurretPVM

# The parameters for the PVM they need to be set the same
# as model that was trained or else an error will be raised
n_color = 3
input_edge_x, input_edge_y = 2, 2
input_size = input_edge_x * input_edge_y * n_color
hidden_size = 8
inner_hidden_size = 8
output_sizes = [0] * 8
inner_output_size = 0
structure = [(64, 48), (32, 24), (16, 12),
             (8, 6), (4, 3), (3, 2), (2, 1), 1]

break_start_x = 16
break_end_x = 49
break_start_y = 12
break_end_y = 37

edge_n_pixels_x, edge_n_pixels_y = (input_edge_x * structure[0][0], 
                                    input_edge_y * structure[0][1])

connect_dict = make_connections(structure, input_size, 
                                hidden_size, output_sizes, 
                                context_from_top_0_0=True)
break_unit_list = []
for x in range(break_start_x, break_end_x):
    for y in range(break_start_y, break_end_y):
        break_unit_list.append('_0_{}_{}'.format(x, y))
connect_dict = break_stuff(connect_dict, break_unit_list, 
                           (input_edge_y, input_edge_x), 
                           inner_hidden_size,
                           inner_output_size)


# dim is a tuple (height, width, number of colors)
dim = (edge_n_pixels_y, edge_n_pixels_x, 3)
input_shape = (input_edge_y, input_edge_x)
basic_index = np.arange(np.prod(dim)).reshape(dim)
flat_map = flatten_image(basic_index, input_shape)
rev_flat_map = np.argsort(flat_map
                          ).reshape(dim)

# Put in your own trained model here
fname = '/home/mhazoglou/PVM_PyCUDA/FullSetTraining/' + \
        'fovea_96h_128w_hidden_8_no_tracking_learning_rate_0.01_3500000steps'

pvm_turret = PhantomXTurretPVM(connect_dict, flat_map, dim, 16, 9, 75,
                               fov_horizontal=True, norm=255., noise=0, 
                               threshold=0.01)

pvm_turret.load_parameters(fname)
print('PVM instance initialized')

# host and port parameters
HOST=''
PORT=8089
buffer = 4096

# socket open
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    print('Socket created')

    s.bind((HOST, PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn, addr = s.accept()
    
    # sending data over connection
    with conn:
        data = "".encode()
        payload_size = struct.calcsize("L")
        
        # finding size of the serialization
        while True:
            while len(data) < payload_size:
                data += conn.recv(buffer)
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            
            # unpacking message size
            msg_size = struct.unpack("L", packed_msg_size)[0]
            
            # receiving data until complete
            while len(data) < msg_size:
                data += conn.recv(buffer)
            frame_data = data[:msg_size]
            data = data[msg_size:]
            
            # deserializing pickled data
            frame = pickle.loads(frame_data)

            print('frame received\n')

            pvm_turret.forward(np.array(frame))
            pvm_turret.evolve()

            print('frame processed\n')

            pan_tilt_values = '{},{}\n'.format(pvm_turret.pan,
                                              pvm_turret.tilt)

            pred_frame = np.array(pvm_turret.norm *
                                  pvm_turret.pred[
                                  :pvm_turret.L_input].get(),
                                  dtype=np.uint8)[rev_flat_map]
            err_frame = np.array(
                pvm_turret.norm * abs(
                    pvm_turret.err[:pvm_turret.L_input].get() - 0.5
                ),
                dtype=np.uint8
            )[rev_flat_map]
            y_shift_center, x_shift_center = int(pvm_turret.y_shift +
                                                 pvm_turret.height / 2), \
                                             int(pvm_turret.x_shift +
                                                 pvm_turret.width / 2)
            err_frame[y_shift_center, x_shift_center, 0] = 0
            err_frame[y_shift_center, x_shift_center, 1] = 0
            err_frame[y_shift_center, x_shift_center, 2] = 255

            send_data = pickle.dumps((pan_tilt_values.encode(),
                                      pred_frame,
                                      err_frame))
            conn.sendall(struct.pack("L", len(send_data)) + send_data)

            print('Pan-Tilt position sent: ' + pan_tilt_values + '\n'
                  + 'Prediction and Prediction error sent\n')


