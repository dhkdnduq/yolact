# import onnx
# Load the ONNX model
# model = onnx.load("yolact-sim.onnx")
# Check that the IR is well formed
# onnx.checker.check_model(model)
# Print a human readable representation of the graph
# print(onnx.helper.printable_graph(model.graph))

# input_file_path = "/media/ernestas/D1/workdir/yolact/data/yolact_example_0.png"
input_file_path = "/media/ernestas/D1/workdir/yolact/data/rand.jpg"
HEIGHT = 550
WIDTH = 550

from utils.augmentations import FastBaseTransform
import cv2
import numpy as np
import torch
import onnxruntime as rt
import time

# img = cv2.imread(input_file_path)
# dim = (WIDTH, HEIGHT)
# img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# frame = torch.from_numpy(img).float()
# frame = np.expand_dims(frame, axis=0)

img = cv2.imread(input_file_path)
print(img.shape)
dim = (WIDTH, HEIGHT)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
print(img.shape)
img = np.expand_dims(img, axis=0)
print(img.shape)
img = np.transpose(img, (0,3,2,1))
print(img.shape)

sess = rt.InferenceSession("yolact.onnx")
# sess.set_providers(['TensorrtExecutionProvider'])
input_name = sess.get_inputs()[0].name
loc_name = sess.get_outputs()[0].name
conf_name = sess.get_outputs()[1].name
mask_name = sess.get_outputs()[2].name
priors_name = sess.get_outputs()[3].name
proto_name = sess.get_outputs()[4].name

frame = torch.from_numpy(cv2.imread(path)).float()
batch = FastBaseTransform()(frame.unsqueeze(0))

# start = time.time()    
# for i in range(100):
pred_onx = sess.run([loc_name, conf_name, mask_name, priors_name, proto_name], {input_name: batch.cpu().detach().numpy()})
# print(f"avg time {(time.time() - start) / (i+1)}")

# detect = Detect(81, bkg_label=0, top_k=200, conf_thresh=0.05, nms_thresh=0.5)
# preds = detect({'loc': torch.from_numpy(pred_onx[0]), 
#                 'conf': torch.from_numpy(pred_onx[1]), 
#                 'mask': torch.from_numpy(pred_onx[2]), 
#                 'priors': torch.from_numpy(pred_onx[3]), 
#                 'proto': torch.from_numpy(pred_onx[4])}, None)

for i,pred in enumerate(pred_onx):
    np.save(str(i), pred)

# print(pred_onx)
# img = cv2.imread(input_file_path)
# print(img.shape)
# dim = (WIDTH, HEIGHT)
# img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# print(img.shape)
# img = np.expand_dims(img, axis=0)
# print(img.shape)
# img = np.transpose(img, (0,3,2,1))
# print(img.shape)

# sess = rt.InferenceSession("yolact.onnx")
# input_name = sess.get_inputs()[0].name
# loc_name = sess.get_outputs()[0].name
# conf_name = sess.get_outputs()[1].name
# mask_name = sess.get_outputs()[2].name
# priors_name = sess.get_outputs()[3].name
# proto_name = sess.get_outputs()[4].name
# # X_test = np.random.rand(1,3,550,550)
# input_file_path = "data/yolact_example_0.png"
# X_test = cv2.imread(input_file_path)
# X_test = np.expand_dims(X_test, axis=0)
# pred = sess.run([loc_name,
#                 conf_name,
#                 mask_name,
#                 priors_name,
#                 proto_name], {input_name: img.astype(np.float32)})[0]

# print(pred[0].shape)