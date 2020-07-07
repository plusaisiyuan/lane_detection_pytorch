#!/bin/bash

set -e

CONDA_EXEC=$HOME/anaconda3/bin
if [ ! -d $CONDA_EXEC ]; then
    echo "---------------------"
    echo "Downloaing miniconda!"
    echo "---------------------"

    echo $CONDA_EXEC
    PACKAGE=Anaconda3-5.2.0-Linux-x86_64.sh
    if [ ! -f $PACKAGE ]; then
      wget https://repo.anaconda.com/miniconda/Anaconda3-5.2.0-Linux-x86_64.sh
    fi
    # This for CI. run `bash Anaconda3-5.2.0-Linux-x86_64.sh` on your local machine for more customized options
    bash Anaconda3-5.2.0-Linux-x86_64.sh -b
fi

#eval "$($CONDA_EXEC/conda shell.bash hook)"

# You can comment the following line, recommended only if you think you know what you are doing
#conda config --set always_yes yes

source $CONDA_EXEC/activate

venv=.venv
VENV_PATH=$HOME/anaconda3/envs/$venv

if [ ! -d $VENV_PATH ]; then
    echo "-----------------------"
    echo "Installing environment"
    echo "-----------------------"

    # Install the environment
    conda env create -f environment.yaml
    
    echo "-----------------------"
    echo "Activate the environment"
    echo "-----------------------"
    
    # Activate the environment
    conda activate $venv

    echo "-----------------------"
    echo "Install the requirements"
    echo "-----------------------"
    
    # Install the requirements
    pip install -r requirements.txt --extra-index-url http://dist:3141/plusai/stable/ --trusted-host dist

    cp -r /usr/lib/python3.6/dist-packages/tensorrt* $VENV_PATH/lib/python3.6/site-packages/
fi

echo "---------"
echo "All done!"
echo "---------"