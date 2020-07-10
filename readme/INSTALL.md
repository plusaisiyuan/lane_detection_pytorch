# Installation


The code was tested on Ubuntu 18.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6.9 and [PyTorch]((http://pytorch.org/)) v1.5.1. NVIDIA GPUs are needed for both training and testing.
After install Anaconda:

0. [Optional but recommended] create a new conda environment. 

    ~~~
    conda create --name lane python=3.6.9
    ~~~
    And activate the environment.
    
    ~~~
    conda activate lane
    ~~~

1. Install pytorch1.5.1:

    ~~~
    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
    ~~~

2. Clone this repo:

    ~~~
    LaneNet_ROOT=~/workspace/lane_detection_pytorch
    git clone git@github.com:NUST-ZHIHAO/lane_detection_pytorch.git $LaneNet_ROOT
    ~~~


3. Install the requirements

    ~~~
    pip install -r $LaneNet_ROOT/requirements.txt
    ~~~