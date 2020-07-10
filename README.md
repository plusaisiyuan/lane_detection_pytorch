# lane_detection_pytorch
We use this multi-task model for our lane detection task.

## Installation
#### Easy Method:
~~~
bash setup.sh
~~~
And then follow the commands at the end.
#### Adventurous:
Please refer to [INSTALL.md](readme/INSTALL.md).

## Use LaneNet

### Train model

~~~
sh scripts/train.sh
~~~

Currently, we can the erfnet as our backbone for lane detection.

### Test model

~~~
python src/demo.py --demo image_dir --arch dlav0_34 --load_model exp/plusai/plusai_dlav0/model_last.pth
~~~

### Evaluate model

~~~
python src/test.py --arch dlav0_34 --test_data /data/obstacle/us --test_label us_test_1.1.json --load_model exp/plusai/plusai_dlav0/model_last.pth
~~~

### Deploy model

#### Convert to onnx

~~~
sh scripts/convert_to_onnx.sh
~~~

#### Convert to tensorrt

~~~
sh scripts/convert_to_tensorrt.sh
~~~

This cmd will generate onnx model file which can be used by our drive repo.

