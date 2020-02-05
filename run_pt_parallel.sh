git pull
cd pt.darts || exit
apt install graphviz
pip install graphviz
python search.py --name cifar10-mg --dataset cifar10 --gpus all \
    --batch_size 64 --workers 16 --print_freq 10 \
    --w_lr 0.1 --w_lr_min 0.004 --alpha_lr 0.0012
