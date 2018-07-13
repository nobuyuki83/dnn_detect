import torch

import cv2

import glob,os,platform
import random,shutil,time

import detect



def evaluate_training_data(net_d, net_c, net_l):

    list_path_csv = []
    if os.uname()[1] == "nubuntu":
#		aPath = glob.glob('/media/nobuyuki/D/projects/face/haal/*_haalpos.txt')	+ aPath
        list_path_csv = glob.glob('/media/nobuyuki/D/projects/face/rect_cv/*.csv') + list_path_csv
    else:
        list_path_csv = glob.glob("/Volumes/PortableHDD/face/rect_cv/*.csv") + list_path_csv
#        list_path_csv = glob.glob("/Volumes/PortableHDD/face/rect_manual/*.csv") + list_path_csv

    for itr in range(len(list_path_csv)):
        path = list_path_csv[random.randint(0,len(list_path_csv)-1)]
        np_img = face.get_imgage_train(path)
        rect_train = face.get_rect_train(path,"rect_face_cv")
        ####
        aRect = face.detect_face_dnn_multires(net_d, net_c, net_l, np_img)
        for rect in aRect:
            face.highlight_rect(np_img, rect, (255, 255, 0), width=1)
        aRectC = face.marge_rects(aRect)
        for rect in aRectC:
            face.highlight_rect(np_img, rect, (0, 255, 255), width=2)
        face.highlight_rect(np_img, rect_train, (0, 0, 255), width=2)
#		print(aRect)
        cv2.imshow('img',np_img)
        if len(aRect) == 0:
            cv2.waitKey(1000)
        if len(aRectC) >= 2:
            cv2.waitKey(2000)
        else:
            cv2.waitKey(300)


def select_good_and_bad(net_d, net_c, net_l, path_dir_root):
    print("select good and bad:", path_dir_root)
    list_path_img = []
    list_path_img = glob.glob(path_dir_root + '/xinbox/*.jpg', recursive=True) + list_path_img
    list_path_img = glob.glob(path_dir_root + '/xinbox/*.png', recursive=True) + list_path_img
    print("there is ",len(list_path_img),"images")
    for itr in range(len(list_path_img)):
        path_img = list_path_img[random.randint(0,len(list_path_img)-1)]
        if not os.path.isfile(path_img):
            continue
        npImg = cv2.imread(path_img)
        if npImg is None:
            continue
        ####
        print(npImg.shape[0],npImg.shape[1])
        list_rect_dnn0 = face.detect_face_dnn_multires(net_d, net_c, net_l, npImg, threshold_prob=0.6)
        for rect in list_rect_dnn0:
            face.highlight_rect(npImg, rect, (255, 255, 0), width=1)
        list_rect_dnn = face.marge_rects(list_rect_dnn0)
        for rect in list_rect_dnn:
            face.highlight_rect(npImg, rect, (0, 255, 255), width=2)
        cv2.imshow('img',npImg)
        key = cv2.waitKey(10000)
        ikey = int(key)
        print(ikey)
        if ikey == 113:
            exit()

        if key == 8: # delete key
            path_img_new = path_dir_root + '/xtrash/' + os.path.basename(path_img)
            print("move_to_trash")
            shutil.move(path_img, path_img_new)
            continue

        if ikey != 32: #space key
            print("this image is bad. do nothing")
            continue

        ### md5
        bnwe_md5 = face.getMD5(path_img)
        ext = path_img.rsplit(".",1)[1]
        dir_dist = path_dir_root+"/"+str(bnwe_md5[0])
        path_img_new = dir_dist+"/"+bnwe_md5+"."+ext
        print(path_img_new,dir_dist)

        if os.path.isfile(path_img_new):
            print("there is already one:")
            continue

        ### register image
        print("found new iamge:",path_img,path_img_new)

        shutil.move(path_img,path_img_new)

        dict_data = {}
        dict_data["url_name"] = os.path.basename(path_img)
        dict_data["shape_img"] = str(npImg.shape[0]) + "," +str(npImg.shape[1])
        if len(list_rect_dnn) > 0:
            str_csv = face.get_str_csv_list_rect(list_rect_dnn)
            dict_data["rect_face_dnn"] = str_csv
        face.save_csv_data(dir_dist+"/"+bnwe_md5+".csv",dict_data)


def main():
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        print("using GPU")
    else:
        print("using CPU")

    net_d,net_c,net_l = face.load_detection_network(is_cuda,'.')

    if len(os.sys.argv) == 2:
        path = os.sys.argv[1]
        select_good_and_bad(net_d, net_c, net_l, path)
    else:
        evaluate_training_data(net_d, net_c, net_l,)



if __name__ == "__main__":
    main()









