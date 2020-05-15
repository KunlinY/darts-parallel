git pull
apt-get update
apt-get --assume-yes install graphviz ufw
pip install graphviz pydot
pip install --upgrade torch torchvision
ufw allow 29500/tcp
python rpc_parameter_server.py --world_size="$1" --rank=0 --num_gpus=1
