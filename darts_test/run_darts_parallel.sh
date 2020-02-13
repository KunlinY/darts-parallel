git pull
#cd pt.darts || exit
apt-get update
apt-get --assume-yes install graphviz
pip install graphviz pydot
cd cnn && python train_search.py --gpu 0,1
