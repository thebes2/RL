#!/bin/bash

sudo apt-get install python3
sudo apt-get install python3-venv
sudo apt-get install python3-pip

mkdir -p venv
python3 -m venv ./venv

source ./venv/bin/activate
pip install wheel
pip install -r requirements.txt

./scripts/run
