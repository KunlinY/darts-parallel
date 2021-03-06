git pull
#cd pt.darts || exit
apt-get update
apt-get --assume-yes install graphviz
pip install graphviz pydot
python search.py --name cifar10-mg --dataset cifar10 --gpus all \
    --batch_size 128 --workers 8 --print_freq 1 \
    --w_lr 0.1 --w_lr_min 0.004 --alpha_lr 0.0012 --noise True
