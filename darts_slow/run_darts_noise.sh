git pull
apt-get update
apt-get --assume-yes install graphviz
pip install graphviz pydot
python search.py --name cifar10-mg --dataset cifar10 --gpus all \
    --noise True --workers "$1"
