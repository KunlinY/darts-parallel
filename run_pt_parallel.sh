git pull
cd pt.darts || exit
pip install graphviz
ipcs -lm
python search.py --name cifar10-mg --dataset cifar10 --gpus all \
    --batch_size 256 --workers 4 --print_freq 10 \
    --w_lr 0.1 --w_lr_min 0.004 --alpha_lr 0.0012
