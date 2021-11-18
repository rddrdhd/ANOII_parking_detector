from MyNet import MyNet

if __name__ == '__main__':
	from nets.GoogLeNet import Inception
	net = MyNet(net_type="GoogLeNet", dimensions=3, epoch=1)
	net.train(img_size=96)  # TODO visual=True)
	net.test(img_size=96)

