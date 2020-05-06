git pull
apt-get update
apt-get --assume-yes install graphviz
pip install graphviz pydot
python augment.py --name cifar10 --dataset cifar10 --gpus all \
    --batch_size "$1" --workers "$2" --genotype "$3" \
    --print_freq 10 --lr 0.1
