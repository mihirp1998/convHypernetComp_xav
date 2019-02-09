# convHypernetComp

How to run
```bash
python -u train.py -f /home_01/f20150198/datasets/ActivityNet/Crawler/Kinetics -N 4 --iterations 16
```
Important files

*train.py* --> Main file to run for training

*network.py* --> Consists of Hypernet, Encoder and Decoder Cell

*utils.py* --> consists of batch wise convolution
