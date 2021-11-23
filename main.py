from MyNet import MyNet
from nets.GoogLeNet import Inception

if __name__ == '__main__':
	#net = MyNet(net_type="GoogLeNet", dimensions=3, epoch=3, batch_size=8, img_size=96 ) # TODO error when dimensions=1
	#net = MyNet(net_type="ResNet", dimensions=3, epoch=3, batch_size=8, img_size=96 ) # TODO error when dimensions=1
	net = MyNet(net_type="DenseNet", dimensions=3, epoch=3, batch_size=8, img_size=96 ) # TODO error when dimensions=1
	#net = MyNet(net_type="VGGNet", dimensions=3, epoch=3, batch_size=8, img_size=96 ) # TODO error when dimensions=1
	net.train()  # TODO visual=True)
	#net.test()
	net.cheat_test(250)

