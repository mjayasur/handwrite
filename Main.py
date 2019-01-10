from MNISTreader import MnistRead
from Network import Network


data = MnistRead("mnist/train/images.idx3-ubyte", "mnist/train/labels.idx1-ubyte", "mnist/test/images.idx3-ubyte", "mnist/test/labels.idx1-ubyte")
vern = Network([784, 30, 10])
vern.train(list(data.test), 10, 50)

