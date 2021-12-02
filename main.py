from MyNet import MyNet
from nets.GoogLeNet import Inception
import torch

if __name__ == '__main__':
	net = MyNet(net_type="GoogLeNet", dimensions=3, epoch=3, batch_size=8, img_size=224)#, pretrained=True )
	#net = MyNet(net_type="ResNet")
	#net = MyNet(net_type="DenseNet")
	#net = MyNet(net_type="VGGNet")
	#net.train()
	#net.test()
	net.cheat_test(threshold=5)  # max 5 white pixels after canny detector to indicate "black night spot without car"

