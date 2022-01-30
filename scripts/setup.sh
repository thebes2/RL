#!/bin/bash

sudo apt-get install python3.8
sudo apt-get install python3-venv
sudo apt-get install python3-pip

pip3 install virtualenv

mkdir -p venv
python3 -m virtualenv --python=`which python3.8` ./venv

source ./venv/bin/activate
pip install wheel
pip install -r requirements.txt

./scripts/run
