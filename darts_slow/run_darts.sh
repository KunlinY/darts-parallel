git pull
apt-get update
apt-get --assume-yes install graphviz
pip install graphviz pydot
python search.py --name cifar10-mg --dataset cifar10 --gpus all \
    --batch_size "$1" --workers "$2" --noise "$3" \
    --print_freq 10 --w_lr 0.1 --w_lr_min 0.004 --alpha_lr 0.0012
