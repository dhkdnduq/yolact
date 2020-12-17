### YOLACTv1 ONNX runtime with TRT execution provider

##### Prerequisites
- Pytorch (tested with 1.7.0)
- YOLACT requirements
  - ``pip3 install -r requirements.txt``
- onnxruntime with TRT execution provider
  - follow detailed instructions: https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#tensorrt
  - one possible issue i see right away, that this build is tested with CUDA 11.0 and cuDNN 8.0. But Jetpack 4.4 supports CUDA 10.0.326 and cuDNN 7.6.3. There could be some trouble building with older versions.
  - i succeded with CUDA 11.1 and cuDNN 8.0
  - add ``--parallel`` for faster build
  - onnxruntime build command: ``./build.sh --parallel --cudnn_home <path to cuDNN e.g. /usr/lib/x86_64-linux-gnu/> --cuda_home <path to folder for CUDA e.g. /usr/local/cuda-11.1> --use_tensorrt --tensorrt_home <path to TensorRT home>``
    - if you get build error complaining about onnx-tensorrt package change ``return std::move(fields);`` to ``return fields;`` in ``builtin_op_importers.cpp``
  - python wheel build command: ``./build.sh --update --build --parallel --build_wheel --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu --tensorrt_home /usr/lib/aarch64-linux-gnu``
    - don't forget to change local paths :)
  - copy built python package to the yolact repo later or add ``<ONNXRuntimeSRCDirectory>/build/Linux/Debug/`` to your PYTHONPATH env variable
    - ``export PYTHONPATH=<ONNXRuntimeSRCDirectory>/build/Linux/Debug:$PYTHONPATH``

##### Execution
- Clone this repo ``onnx`` branch
  - ``git clone -b onnx https://github.com/simutisernestas/yolact`` 
- Download model weights from original repo README https://github.com/dbolya/yolact
  - Tested only with Resnet50-FPN (yolact_resnet50_54_800000.pth)
- Run ``export_onnx.sh`` bash script to export onnx model
  - ``chmod +x export_onnx.sh``
  - ``./export_onnx.sh``
  - This will produce ``yolact.onnx`` file
- Run ``lab.py`` script (loading of model will take some time)
  - ``python3 lab.py``
- If you get error: "Cuda out of memory"
  - set ORT_TENSORRT_MAX_WORKSPACE_SIZE to smaller value
    - ``export ORT_TENSORRT_MAX_WORKSPACE_SIZE=500000000`` ~500MB
  - read more about tunning: https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/TensorRT-ExecutionProvider.md
- Check results directory

##### Nvidia Jetson TX1/TX2/Nano/Xavier
  - Tested on JetPack SDK 4.4 with CUDA 10.2 and Pytorch 1.7.0
  - On Nano, follow this [section](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#nvidia-jetson-tx1tx2nanoxavier) of the onnxruntime instruction to build, this would take a while
  - Clone this repo to Nano
  - Download the pre exported .onnx [here](https://drive.google.com/file/d/1fAFd0tKSFTy2Id6zhoEBfKxuNgo73XEN/view?usp=sharing) (exporting this model on Nano should take a long time)
  - Run the `lab.py` script
  
