#!/usr/bin/python
import sys
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob
GAUSS_SIZE = (5, 5)

MAX_BRIGHTNESS = 255
COLOR_RED = (0, 0, MAX_BRIGHTNESS)
COLOR_GREEN = (0, MAX_BRIGHTNESS, 0)


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


def get_true_results(filename='groundtruth.txt'):
    # naštu do pole výsledky - 1=auto, 0=prázdné
    test_file = open(filename, 'r')
    test_lines = test_file.readlines()
    test_values = []

    for line in test_lines:
        st_line = line.strip()
        test_values.append(st_line)

    return test_values


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


def get_dictionary_mean(dict_list):
    # zprůměrovat skóre napříč všemi obrázky parkovišť
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(float(d[key]) for d in dict_list) / len(dict_list)
    return mean_dict


def get_parking_evaluation(TP, TN, FP, FN, i):
    # vyhodnotím všechny parkovací místa z obrázku
    precision = float(float(TP)/(TP+FP))
    sensitivity = float(float(TP)/(TP+FN))

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "sensitivity": sensitivity,
        # f1 score - harmonic mean of precision and sensitivity
        "f1": 2*((precision*sensitivity)/(precision+sensitivity)),
        "accuracy": (float)(TP+TN)/(float)(i)
    }


def print_evaluation_header():
    print("TP\tTN\tFP\tFN\tprecision\tsensitivity\tf1\t\taccuracy")


def print_evaluation_result(result):
    print("{:.0f}".format(result.get("TP")), end="\t")
    print("{:.0f}".format(result.get("TN")), end="\t")
    print("{:.0f}".format(result.get("FP")), end="\t")
    print("{:.0f}".format(result.get("FN")), end="\t")
    print("{:.4f}".format(result.get("precision")), end="\t\t")
    print("{:.4f}".format(result.get("sensitivity")), end="\t\t")
    print("{:.4f}".format(result.get("f1")), end="\t\t")
    print("{:.4f}".format(result.get("accuracy")))


def main(argv):
    # vytvořím okno
    cv2.namedWindow("blur_image", 0)
    # cv2.namedWindow("res_image", 0)
    # cv2.namedWindow("edge_image", 0)

    # načtu výsledky
    true_results = get_true_results()

    # nachystám si struktury/počítadla
    my_results = []

    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0
    results_evaluation = []
    iii = 0

    # projdu je přes testovací obrázky
    test_images = [img for img in glob.glob("test_images/*.jpg")]
    test_images.sort()

    # body rohů parkovišť
    parking_lot_coordinates = get_coordinates()

# TRAIN
    # načtu si obrázky prázdných parkovacích míst
    train_images_free = [img for img in glob.glob(
        "train_images/free/*.png")]
    # načtu si obrázky plných parkovacích míst
    train_images_full = [img for img in glob.glob(
        "train_images/full/*.png")]

    train_labels_list = []  # nuly nebo jednicky
    train_images_list = []  # obrazky
    IMG_SIZE = 96  # default je pro člověka je 64x128, my chceme čberec na parkovací místo

   # help(cv2.HOGDescriptor)

    # obrazek se rozdeli do mensich, pak se spocita hog, a pak se zas spoji
    # TODO menit abych dosahla co nejlepsich vysledku
    hog = cv2.HOGDescriptor((IMG_SIZE, IMG_SIZE), (32, 32),
                            (16, 16), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)

    # klasifikator - tady svm, ale muze byt treba k nejblizsich sousedu
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)  # urcim si typ
    svm.setKernel(cv2.ml.SVM_INTER)  # urcim si kernel
    svm.setC(100.0)  # nastavim si hodnotu C

    # kdy se ma trenovani zastavit - maximalni pocet iteraci
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))

# hog trenovani
    for img_num in range(len(train_images_full)):  # jdu pres plne trenovaci obrazky
        one_park_image = cv2.imread(
            train_images_full[img_num], 0)  # nactu misto cernobile
        res_image = cv2.resize(one_park_image, (IMG_SIZE, IMG_SIZE))  # resiznu

        # spocitam histogram orientovanych gradientu
        hog_feature = hog.compute(res_image)
        train_images_list.append(hog_feature)
        train_labels_list.append(1)

    for img_num in range(len(train_images_free)):  # jdu pres plne trenovaci obrazky
        one_park_image = cv2.imread(
            train_images_free[img_num], 0)  # nactu misto cernobile
        res_image = cv2.resize(one_park_image, (IMG_SIZE, IMG_SIZE))  # resiznu

        hog_feature = hog.compute(res_image)  # spocitam HOG
        train_images_list.append(hog_feature)
        train_labels_list.append(0)

    svm.train(np.array(train_images_list),
              cv2.ml.ROW_SAMPLE, np.array(train_labels_list))
    svm.save("my_det.xml")

    print("TRAIN HOG DONE on ", len(train_labels_list), " images")
# TESTING

    for img in test_images:
        one_park_image = cv2.imread(img)
        one_park_image_draw = cv2.imread(img)

        for parking_spot_coordinates in parking_lot_coordinates:

            parking_spot_coordinates_float = get_points_float(
                parking_spot_coordinates)
            # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
            # vyrovnám a vyřežu parkovací místo podle bodů
            warped_image = four_point_transform(
                one_park_image, np.array(parking_spot_coordinates_float))
            # resize narovnaného obrazu
            res_image = cv2.resize(warped_image, (IMG_SIZE, IMG_SIZE))
            # vymažu šum
            blur_image = cv2.GaussianBlur(res_image, GAUSS_SIZE, 0)
            # převedu do černobílého
            gray_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
# Predict label
            hog_feature = hog.compute(gray_image)
            predict_label = svm.predict(np.array(hog_feature).reshape(1, -1))
           # print(predict_label[1])
            if (predict_label[1] == 1.):
                # ZABRANE MISTO
                my_results.append(1)
            else:
                # VOLNE MISTO
                my_results.append(0)

                # POSEM TO FUNGUJE DOBRE DOPICI
# konec vlastni logiky
            # pro vykreslení - dej mi body okolo parkovacího místa
            parking_spot_coordinates_int = get_points_int(
                parking_spot_coordinates)
            # pro pozdější zobrazení jen špatně vyhodnocených parkovacích míst
            bad = False

            # to je [0] v prvnim, [0,0] v druhem, [0, 0, 0, 0, 0, 0, 0, 1] v 8. ...
            # print(program_results)
            # print(len(ground_truth))  # 1334
# TODO
            # vyhodnocení
# ŠPATNĚ VYHODNOCENÍ

            correct_answer = int(true_results[iii])  # TODO
            my_answer = int(my_results[-1])
            #print("correct ans:\t", correct_answer,"\tmy ans:\t", my_answer, "\ti:\t", iii)
            if(correct_answer == 1 and my_answer == 0):
                # parkovaci misto neni volne, ale program detekoval ze ano
                false_negative += 1
                draw_rect(one_park_image_draw,
                          parking_spot_coordinates_int, (255, 0, 0, 100))
                bad = True
            elif(correct_answer == 0 and my_answer == 1):
                # parkovaci misto je volne, ale program detekoval ze ne
                false_positive += 1
                bad = True
                draw_cross(one_park_image_draw,
                           parking_spot_coordinates_int, (255, 0, 0, 100))
            elif(correct_answer == 1 and my_answer == 1):
                # parkovaci misto neni volne, program detekoval spravne
                true_positive += 1
                draw_cross(one_park_image_draw, parking_spot_coordinates_int)
            elif(correct_answer == 0 and my_answer == 0):
                # parkovaci misto je volne, program detekoval spravne
                true_negative += 1
                draw_rect(one_park_image_draw, parking_spot_coordinates_int)
            iii += 1

            # zobrazím obrázky jen pokud se špatně vyhodnotily
            if(bad):
                # cv2.imshow('blur_image', blur_image)
                cv2.imshow('res_image', res_image)
               # cv2.imshow('edge_image', edge_image)
                cv2.waitKey(0)
            # roi = img[y:y+h, x:x+w]

        evaluation_result = get_parking_evaluation(
            true_positive, true_negative, false_positive, false_negative, iii)

        results_evaluation.append(evaluation_result)
        cv2.imshow('one_park_image', one_park_image_draw)

        key = cv2.waitKey(0)
        if key == 27:  # exit on ESC
            break
    avg_evaluation_result = get_dictionary_mean(results_evaluation)
    print("SETTINGS:\tgauss:", GAUSS_SIZE)
    print_evaluation_header()
    print_evaluation_result(avg_evaluation_result)


if __name__ == "__main__":
    main(sys.argv[1:])
