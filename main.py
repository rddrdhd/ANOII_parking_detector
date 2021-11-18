from MyNet import MyNet
from nets.GoogLeNet import Inception

if __name__ == '__main__':
	net = MyNet(net_type="GoogLeNet", dimensions=3, epoch=1, img_size=224 ) # TODO error when dimensions=1
	net.train()  # TODO visual=True)
	net.test()

