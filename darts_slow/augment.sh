git pull
apt-get update
apt-get --assume-yes install graphviz
pip install graphviz pydot
python augment.py --name cifar10 --dataset cifar10 \
    --batch_size 64 --workers "$1" --genotype "$2"
