
import cv2
import numpy as np
import glob
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms.functional as TF
from nets.GoogLeNet import GoogLeNet
from nets.VGGNet import VGGNet
from PIL import Image
import utils
#from torchvision.models.googlenet import googlenet


def is_it_empty_in_night(canny_image, treshold):
    n_white_pix = np.sum(canny_image == 255)
    if n_white_pix > treshold:
        print('Auto! Má totiž tolik pixelů hran:', n_white_pix)
        return False
    else:
        print('Prádné! Má totiž tolik pixelů hran:', n_white_pix)
        return True


class MyNet:

    def __init__(self, dimensions, net_type, batch_size=8, epoch=1, img_size = 224):
        self.data = []
        #TODO fix error for grayscale 
        self.dimensions = dimensions  # grayscale=1 / rgb=3
        self.type = net_type
        self.batch_size = batch_size
        self.epoch = epoch
        self.img_size = img_size

        self.path = 'trained_nets/'+net_type+'_e'+str(epoch)+'_d'+str(dimensions)+'_s'+str(self.img_size)+'.pth'
        self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.7, hue=0),
            ])
        """
        if(self.dimensions == 1): # grayscale img
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=3)
            ])
        else: # rgb img
             self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                #transforms.ColorJitter(brightness=1.5, contrast=1, saturation=0.7, hue=0),
                
            ])
        """

    def train(self):

        # prepair data
        data_dir = 'train_images'
        image_datasets = datasets.ImageFolder(data_dir, transform=self.transform)
        data_loader = torch.utils.data.DataLoader(
            image_datasets, batch_size=self.batch_size, shuffle=True, num_workers=4)
        classes = ('free', 'full')
        images, labels = iter(data_loader).next()

        #print(' '.join('%5s' % classes[labels[j]] for j in range(self.batch_size)))
        #cv2.imshow(torchvision.utils.make_grid(images))
        utils.imshow(torchvision.utils.make_grid(images))
        # net types
        if(self.type == "GoogLeNet"):
            net = GoogLeNet(3).net  # models.googlenet(pretrained=True)

        elif(self.type == "VGGNet"):
            net = VGGNet(3).net  # models.googlenet(pretrained=True)

  

        # using CUDA if availble
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using",device)
        net.to(device)

        # magic, do not touch
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        print("Starting trainig:\t",self.type,"with",self.epoch,"epochs and",self.dimensions,"dimensions")
        # training, yay!
        for epoch in range(self.epoch):  # loop over the dataset multiple times
            print('epoch %d' % epoch)
            running_loss = 0.0
            for i, data in enumerate(data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

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

        print('training finished', end="\t")
        torch.save(net, self.path)
        print(self.type, 'saved to', self.path)

    def test(self):
       
        actual_results = utils.get_true_results()  # ground truth

        predicted_results = []  # net results
        iii = 0  # iterator
        false_positive = 0  # false positive
        false_negative = 0  # false negative
        true_positive = 0  # true positive
        true_negative = 0  # true negative 

        net = torch.load(self.path)

        print(self.type,"loaded from",self.path)

        net.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available()
                              else "cpu")  # pokud mam cuda device
        net.to(device)  # prepni se sit na cudu/cpu
        test_images = [img for img in glob.glob("test_images/*.jpg")]
        test_images.sort()

        parking_lot_coordinates = utils.get_coordinates()

        print("Starting testing")

        for img in test_images:
            one_park_image = cv2.imread(img)
            one_park_image_show = one_park_image.copy()
            #TF.solarize(img, 0.255)
            

            for parking_spot_coordinates in parking_lot_coordinates:

                pts_float = utils.get_points_float(parking_spot_coordinates)
                pts_int = utils.get_points_int(parking_spot_coordinates)
                warped_image = utils.four_point_transform(
                    one_park_image, np.array(pts_float))
                res_image = cv2.resize(warped_image, (self.img_size, self.img_size))
                
                
                #new_img = transforms.functional.equalize(res_image)#transforms.functional.adjust_brightness(res_image,brightness_factor=1.5)
                one__img = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(one__img)

                image_pytorch = self.transform(img_pil).to(device)
                image_pytorch = image_pytorch.unsqueeze(0)
                output_pytorch = net(image_pytorch)

                _, predicted = torch.max(output_pytorch, 1)
                spotted_car = predicted[0]
                predicted_results.append(spotted_car)
                #print("parking spot coord:", end="\t")
                if(actual_results[iii] and spotted_car):
                    true_positive += 1
                    utils.draw_cross(one_park_image_show, pts_int)
                    #print("TP")
                if(actual_results[iii] and not spotted_car):
                    false_negative += 1
                    utils.draw_rect(one_park_image_show, pts_int, utils.COLOR_BLUE)
                    #print("FN")
                if(not actual_results[iii] and spotted_car):
                    false_positive += 1
                    utils.draw_cross(one_park_image_show, pts_int, utils.COLOR_BLUE)
                    #print("FP")
                if(not actual_results[iii] and not spotted_car):
                    true_negative += 1
                    utils.draw_rect(one_park_image_show, pts_int)
                    #print("TN")

                iii += 1

            cv2.imshow('one_park_image', one_park_image_show)

            key = cv2.waitKey(0)
            if key == 27:  # exit on ESC
                break
         
        print("Testing finished for \t",self.type,"with",self.epoch,"epochs and",self.dimensions,"dimensions")
       
        eval_result = utils.get_parking_evaluation(
            true_positive, true_negative, false_positive, false_negative, iii)
        utils.print_evaluation_header()
        utils.print_evaluation_result(eval_result)


    def cheat_test(self, treshold):

        actual_results = utils.get_true_results()  # ground truth

        predicted_results = []  # net results
        iii = 0  # iterator
        false_positive = 0  # false positive
        false_negative = 0  # false negative
        true_positive = 0  # true positive
        true_negative = 0  # true negative

        net = torch.load(self.path)

        print(self.type, "loaded from", self.path)

        net.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available()
                              else "cpu")  # pokud mam cuda device
        net.to(device)  # prepni se sit na cudu/cpu
        test_images = [img for img in glob.glob("test_images/*.jpg")]
        test_images.sort()

        parking_lot_coordinates = utils.get_coordinates()

        print("Starting testing")

        for img in test_images:
            one_park_image = cv2.imread(img)
            one_park_image_show = one_park_image.copy()
            # TF.solarize(img, 0.255)

            for parking_spot_coordinates in parking_lot_coordinates:

                pts_float = utils.get_points_float(parking_spot_coordinates)
                pts_int = utils.get_points_int(parking_spot_coordinates)
                warped_image = utils.four_point_transform(
                    one_park_image, np.array(pts_float))
                res_image = cv2.resize(warped_image, (self.img_size, self.img_size))

                blur_image = cv2.GaussianBlur(res_image, (5, 5), 0)
                canny_image = cv2.Canny(blur_image, 40, 120)
                by_canny = False
                if is_it_empty_in_night(canny_image, treshold):
                    spotted_car = 0
                    predicted_results.append(spotted_car)
                    cv2.imshow('canny_image', canny_image)
                    by_canny = True

                else:
                    # new_img = transforms.functional.equalize(res_image)#transforms.functional.adjust_brightness(res_image,brightness_factor=1.5)
                    one__img = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(one__img)

                    image_pytorch = self.transform(img_pil).to(device)
                    image_pytorch = image_pytorch.unsqueeze(0)
                    output_pytorch = net(image_pytorch)

                    _, predicted = torch.max(output_pytorch, 1)
                    spotted_car = predicted[0]
                    predicted_results.append(spotted_car)
                    # print("parking spot coord:", end="\t")
                if (actual_results[iii] and spotted_car):
                    true_positive += 1
                    utils.draw_cross(one_park_image_show, pts_int)

                    # print("TP")
                if (actual_results[iii] and not spotted_car):
                    false_negative += 1
                    if (by_canny):
                        utils.draw_rect(one_park_image_show, pts_int, ((150, 150, 255)))
                    else:
                        utils.draw_rect(one_park_image_show, pts_int, utils.COLOR_BLUE)
                    # print("FN")
                if (not actual_results[iii] and spotted_car):
                    false_positive += 1
                    if (by_canny):
                        utils.draw_cross(one_park_image_show, pts_int, (150, 150, 255))
                    else:
                        utils.draw_cross(one_park_image_show, pts_int, utils.COLOR_BLUE)
                    # print("FP")
                if (not actual_results[iii] and not spotted_car):
                    true_negative += 1
                    utils.draw_rect(one_park_image_show, pts_int)
                    # print("TN")

                iii += 1

            cv2.imshow('one_park_image', one_park_image_show)

            key = cv2.waitKey(0)
            if key == 27:  # exit on ESC
                break

        print("Testing finished for \t", self.type, "with", self.epoch, "epochs and", self.dimensions, "dimensions")

        eval_result = utils.get_parking_evaluation(
            true_positive, true_negative, false_positive, false_negative, iii)
        utils.print_evaluation_header()
        utils.print_evaluation_result(eval_result)
