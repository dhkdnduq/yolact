from utils.augmentations import FastBaseTransform
import cv2
import numpy as np
import torch
import onnxruntime as rt
import time
import layers
import eval
import matplotlib.pyplot as plt
import time 
eval.get_args()
import glob
from pathlib import Path
Path("results").mkdir(exist_ok=True)
import imutils

print("Loading trt execution, this may take a while... ")
sess = rt.InferenceSession("yolact.onnx")
print("Onnx execution prodivers:")
print(sess.get_providers())
print("WARNING! If above list doesn't contain TensorrtExecutionProvider model isn't executed by TensorRT")
input_name = sess.get_inputs()[0].name
loc_name = sess.get_outputs()[0].name
conf_name = sess.get_outputs()[1].name
mask_name = sess.get_outputs()[2].name
priors_name = sess.get_outputs()[3].name
proto_name = sess.get_outputs()[4].name

def inference(img_path, show=False):
    # if too big to fit on GPU
    img = cv2.imread(img_path)
    img = imutils.resize(img, width=1024)
    frame = torch.Tensor(img).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))

    start = time.time()
    for i in range(1):
        # inference benchmark
        pred_onx = sess.run([loc_name, conf_name, mask_name, priors_name, proto_name], {input_name: batch.cpu().detach().numpy()})
        detection = layers.Detect(81, bkg_label=0, top_k=200, conf_thresh=0.05, nms_thresh=0.5)
        preds = detection({'loc': torch.from_numpy(pred_onx[0]), 
                        'conf': torch.from_numpy(pred_onx[1]), 
                        'mask': torch.from_numpy(pred_onx[2]), 
                        'priors': torch.from_numpy(pred_onx[3]),
                        'proto': torch.from_numpy(pred_onx[4])})
        print(f"Average inference duration: {(time.time()-start)/(i+1)}s")

    img = eval.prep_display(preds, frame.cpu(), None, None, None, None, undo_transform=False)
    cv2.imwrite('results/' + img_path.split('/')[1], img)

    if show:
        plt.imshow(img)
        plt.show()

print("Starting inference...")
for f in glob.glob("data/*.jpg"):
    print(f"Processing {f}")
    inference(f)
