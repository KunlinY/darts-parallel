git pull
#cd pt.darts || exit
apt-get update
apt-get --assume-yes install graphviz
pip install graphviz pydot
python search.py --name cifar10 --dataset cifar10 --noise True --workers 4 --print_freq 1
