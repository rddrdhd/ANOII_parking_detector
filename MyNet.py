
import cv2
import numpy as np
import math
import glob
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from nets.GoogLeNet import GoogLeNet
from PIL import Image
#from torchvision.models.googlenet import googlenet

MAX_BRIGHTNESS = 255
COLOR_BLUE = (MAX_BRIGHTNESS, 0, 0)
COLOR_RED = (0, 0, MAX_BRIGHTNESS)
COLOR_GREEN = (0, MAX_BRIGHTNESS, 0)


def draw_rect(img, points, color=COLOR_GREEN):
    # obtáhnout parkovací místo
    cv2.line(img, points[0], points[1], color, 3)
    cv2.line(img, points[1], points[2], color, 3)
    cv2.line(img, points[2], points[3], color, 3)
    cv2.line(img, points[3], points[0], color, 3)


def draw_cross(img, points, color=COLOR_RED):
    # křížek na body parkovacího místa
    cv2.line(img, points[0], points[2], color, 3)
    cv2.line(img, points[3], points[1], color, 3)


def get_true_results(filename='groundtruth.txt'):
    with open(filename) as truth_file:
        truth = [int(x) for x in truth_file.read().splitlines()]
    return truth


def get_points_float(one_c):
    # načtu si body parkovacích míst ve floatu
    pts = [((float(one_c[0])), float(one_c[1])),
           ((float(one_c[2])), float(one_c[3])),
           ((float(one_c[4])), float(one_c[5])),
           ((float(one_c[6])), float(one_c[7]))]
    return pts


def get_points_int(coordinate):
    # načtu si body parkovacích míst v int
    point_1 = (int(coordinate[0]), int(coordinate[1]))
    point_2 = (int(coordinate[2]), int(coordinate[3]))
    point_3 = (int(coordinate[4]), int(coordinate[5]))
    point_4 = (int(coordinate[6]), int(coordinate[7]))
    return [point_1, point_2, point_3, point_4]


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def get_coordinates(filename='parking_map_python.txt'):
    # načtu souřadnice bodů - čtveřice pro každé parkovací místo
    pkm_file = open(filename, 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)
    return pkm_coordinates


def get_parking_evaluation(TP, TN, FP, FN, i):
    # vyhodnotím všechny parkovací místa z obrázku
    precision = float(float(TP)/float(TP+FP))
    sensitivity = float(float(TP)/float(TP+FN))
    F1 = 2.0*(float(precision*sensitivity)/float(precision+sensitivity))
    mcc_sqrt = math.sqrt(float(TP+FP)*float(TP+FN)*float(TN+FP)*float(TN+FN))
    MCC = float(TP*TN - FP*FN)/float(mcc_sqrt)
    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "sensitivity": sensitivity,
        # f1 score - harmonic mean of precision and sensitivity
        "f1": F1,
        "accuracy": (float)(TP+TN)/(float)(i),
        "mcc": MCC
    }


def print_evaluation_header():
    print("TP\tTN\tFP\tFN\tprecision\tsensitivity\tf1\t\taccuracy\tMCC")


def print_evaluation_result(result):
    print("{:.0f}".format(result.get("TP")), end="\t")
    print("{:.0f}".format(result.get("TN")), end="\t")
    print("{:.0f}".format(result.get("FP")), end="\t")
    print("{:.0f}".format(result.get("FN")), end="\t")
    print("{:.4f}".format(result.get("precision")), end="\t\t")
    print("{:.4f}".format(result.get("sensitivity")), end="\t\t")
    print("{:.4f}".format(result.get("f1")), end="\t\t")
    print("{:.4f}".format(result.get("accuracy")), end="\t\t")
    print("{:.4f}".format(result.get("mcc")))


class MyNet:

    def __init__(self, dimensions, net_type, batch_size=8, epoch=1):
        self.data = []
        self.dimensions = dimensions  # grayscale=1 / rgb=3
        self.type = net_type
        self.batch_size = batch_size
        self.path = 'trained_nets/'+net_type+'_e'+str(epoch)+'_d'+str(dimensions)+'.pth'
        self.epoch = epoch

    def train(self, img_size):
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
        if(self.dimensions == 1):
            transform.append(transforms.Grayscale(num_output_channels=1))
        data_dir = 'train_images'
        image_datasets = datasets.ImageFolder(
            data_dir, transform=transform)
        data_loader = torch.utils.data.DataLoader(
            image_datasets, batch_size=self.batch_size, shuffle=True, num_workers=4)
        # print(image_datasets)
        classes = ('free', 'full')
        images, labels = iter(torch.utils.data.DataLoader(
            image_datasets, batch_size=self.batch_size, shuffle=True, num_workers=4)).next()

        print(' '.join('%5s' % classes[labels[j]]
              for j in range(self.batch_size)))

        if(self.type == "GoogLeNet"):
            net = GoogLeNet().net  # models.googlenet(pretrained=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using ",device)
        net.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(self.epoch):  # loop over the dataset multiple times
            print('epoch %d' % epoch)
            running_loss = 0.0
            for i, data in enumerate(data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                #inputs, labels = data[0], data[1]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 20 == 19:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 20))
                    running_loss = 0.0

        print(self.type, ' finished training')
        torch.save(net, self.path)
        print(self.type, ' saved to ', self.path)

    def test(self, img_size):
        actual_results = get_true_results()  # ground truth
        predicted_results = []  # net results
        iii = 0  # iterator
        false_positive = 0  # false positive
        false_negative = 0  # false negative
        true_positive = 0  # true positive
        true_negative = 0  # true negative

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

        net = torch.load(self.path)
        net.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available()
                              else "cpu")  # pokud mam cuda device
        net.to(device)  # prepni se sit na cudu/cpu
        test_images = [img for img in glob.glob("test_images/*.jpg")]
        test_images.sort()

        parking_lot_coordinates = get_coordinates()

        for img in test_images:
            one_park_image = cv2.imread(img)
            one_park_image_show = one_park_image.copy()

            for parking_spot_coordinates in parking_lot_coordinates:

                pts_float = get_points_float(parking_spot_coordinates)
                pts_int = get_points_int(parking_spot_coordinates)
                warped_image = four_point_transform(
                    one_park_image, np.array(pts_float))
                res_image = cv2.resize(warped_image, (img_size, img_size))

                if self.dimensions == 3:
                    one__img = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
                else:
                    one__img = one__img = cv2.cvtColor(
                        res_image, cv2.COLOR_BGR2GRAY)
                img_pil = Image.fromarray(one__img)
                image_pytorch = transform(img_pil).to(device)
                image_pytorch = image_pytorch.unsqueeze(0)
                output_pytorch = net(image_pytorch)

                _, predicted = torch.max(output_pytorch, 1)
                spotted_car = predicted[0]
                predicted_results.append(spotted_car)
                print("parking spot coord:", end="\t")
                if(actual_results[iii] and spotted_car):
                    true_positive += 1
                    draw_cross(one_park_image_show, pts_int)
                    print("TP")
                if(actual_results[iii] and not spotted_car):
                    false_negative += 1
                    draw_rect(one_park_image_show, pts_int, COLOR_BLUE)
                    print("FN")
                if(not actual_results[iii] and spotted_car):
                    false_positive += 1
                    draw_cross(one_park_image_show, pts_int, COLOR_BLUE)
                    print("FP")
                if(not actual_results[iii] and not spotted_car):
                    true_negative += 1
                    draw_rect(one_park_image_show, pts_int)
                    print("TN")

                iii += 1
            print(cv2.__version__)

            print("NEXT PARKNIG LOT")

            cv2.imshow('one_park_image', one_park_image_show)

            key = cv2.waitKey(0)
            if key == 27:  # exit on ESC
                break

        eval_result = get_parking_evaluation(
            true_positive, true_negative, false_positive, false_negative, iii)
        print_evaluation_header()
        print_evaluation_result(eval_result)
