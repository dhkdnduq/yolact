import numpy as np
from PIL import Image
import tensorrt as trt
import skimage.transform

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

MEAN = (71.60167789, 82.09696889, 72.30508881)
CLASSES = 20
HEIGHT = 512
WIDTH = 1024

def sub_mean_chw(data):
   data = data.transpose((1, 2, 0))  # CHW -> HWC
   data -= np.array(MEAN)  # Broadcast subtract
   data = data.transpose((2, 0, 1))  # HWC -> CHW
   return data

def rescale_image(image, output_shape, order=1):
   image = skimage.transform.resize(image, output_shape,
               order=order, preserve_range=True, mode='reflect')
   return image

import engine as eng
import inference as inf
import tensorrt as trt 

input_file_path = "data/yolact_example_0.png"
serialized_plan_fp32 = "my_engine.trt"
HEIGHT = 550
WIDTH = 550

import cv2
img = cv2.imread(input_file_path)
print(img.shape)
dim = (WIDTH, HEIGHT)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
print(img.shape)

engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)
out = inf.do_inference(engine, img, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)