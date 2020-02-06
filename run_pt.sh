git pull
cd pt.darts || exit
apt-get --assume-yes update
apt-get --assume-yes install graphviz
pip install graphviz pydot
python search.py --name cifar10 --dataset cifar10
