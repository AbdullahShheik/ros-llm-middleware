Things to have to run the LLM files.

#put your GROQ API KEY in line #30 of layer1_pipeline.py

#if running on Ubuntu, first-time setup:

sudo apt update
sudo apt install -y python3-venv python3-networkx
cd ~/layer1-project
python3 -m venv --system-site-packages ~/layer1-venv
source ~/layer1-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install groq
source /opt/ros/jazzy/setup.bash

