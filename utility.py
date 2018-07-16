
import glob, os, shutil,random
import cv2
import torch
import detect

def conv1():
    #	path_data =  '/usr/local/lib/python3.6/site-packages/cv2/data/'
    path_data = cv2.data.haarcascades
    face_cascade = cv2.CascadeClassifier(path_data + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(path_data + 'haarcascade_eye.xml')

    path_dir = '/Volumes/PortableHDD/face/png'

    for path in glob.glob(path_dir + "/*"):
        if not path.endswith(".png") and not path.endswith(".jpg"): continue
        bn = os.path.basename(path)
        bnn = bn.rsplit(".", 1)[0]
        path1 = path_dir + "/../png_conf/" + bnn + ".txt"
        if not os.path.isfile(path1): continue
        print(bn, bnn)
        with open(path1, "r") as fin:
            s = fin.read()
            s0 = s.split("\n")[0]
            s1, s2 = s0.split(" ")[:2]

        if not s1 == "bb_head0": continue
        x0, y0, x1, y1 = s2.split("_")[:4]
        x0 = float(x0)
        y0 = float(y0)
        x1 = float(x1)
        y1 = float(y1)
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        lxy = max(x1 - x0, y1 - y0)
        rect = [cx - lxy * 0.5, cy - lxy * 0.5, lxy, lxy]

        path2 = path_dir + "/" + bnn + ".csv"
        with open(path2, "w") as fout:
            fout.write(
                "face_rect," + str(rect[0]) + "," + str(rect[1]) + "," + str(rect[2]) + "," + str(rect[3]) + "\n")

        img = cv2.imread(path)
        detect.highlight_rect(img, rect, (0, 255, 255), 2)

        size = (300, int(float(img.shape[0]) / img.shape[1] * 300.0))
        img2 = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC);
        compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        cv2.imwrite(path_dir + "/" + bnn + "_rect.png", img2, compression_params)

        cv2.imshow('img', img2)
        cv2.waitKey(100)

    exit()


def conv2():
    for path in glob.glob("/Volumes/PortableHDD/face/rect_cv/*_haalpos.txt"):
        path_input = path.rsplit('/', 1)[0]
        bn = os.path.basename(path)
        bn = bn[:bn.rfind("_")]
        rect = []
        with open(path_input + "/" + bn + "_haalpos.txt", 'r') as fin:
            lines = fin.readlines()
            assert len(lines) != 0
            aS = lines[0].split(' ')
            rect = [float(aS[0]), float(aS[1]), float(aS[2]), float(aS[3])]
        with open(path_input + "/" + bn + ".csv", 'w') as fout:
            fout.write(
                "face_rect," + str(rect[0]) + "," + str(rect[1]) + "," + str(rect[2]) + "," + str(rect[3]) + "\n")


# change keyword
def conv3():
    key_old = "face_rect"
    key_new = "rect_face_manual"
    for path_csv in glob.glob("/Volumes/PortableHDD/face/rect_manual/*.csv"):
        #        print(path_csv)
        dict_data = detect.get_csv_data(path_csv)
        detect.swap_keyword(key_old, key_new, dict_data)
        print(dict_data)
        detect.save_csv_data(path_csv, dict_data)


# remove "*_rect.png"
def conv4():
    path_dir = "/Volumes/PortableHDD/face/rect_cv/x"
    for path_rect in glob.glob(path_dir + "/*_rect.png"):
        print(path_rect)
        if os.path.isfile(path_rect):
            os.remove(path_rect)


# make the md5
def conv5():
    path_dir = "/Volumes/PortableHDD/face/rect_manual"
    for path_csv in glob.glob(path_dir + "/*.csv"):
        bn = os.path.basename(path_csv)
        bnwe = bn.rsplit(".", 1)[0]
        path_png = path_dir + "/" + bnwe + ".png"
        if not os.path.isfile(path_png): continue
        bnwe_md5 = getMD5(path_png)
        os.rename(path_dir + "/" + bnwe + ".png", path_dir + "/" + bnwe_md5 + ".png")
        os.rename(path_dir + "/" + bnwe + ".csv", path_dir + "/" + bnwe_md5 + ".csv")


def conv6():
    path_dir_root = "/Volumes/PortableHDD/face/rect_cv"
    hex = [str(format(i, 'x')) for i in range(0, 16)]
    print(hex)
    for ch in hex:
        if not os.path.isdir(path_dir_root + "/" + ch):
            os.mkdir(path_dir_root + "/" + ch)

    for path_csv in glob.glob(path_dir_root + "/**/*.csv", recursive=True):
        bn = os.path.basename(path_csv)
        bnwe = bn.rsplit(".", 1)[0]
        path_dir_img = os.path.dirname(path_csv)
        ext_img = ".png"
        if not os.path.isfile(path_dir_img + "/" + bnwe + ext_img):
            ext_img = ".jpg"
        path_img = path_dir_img + "/" + bnwe + ext_img
        assert os.path.isfile(path_img)
        bnwe_md5 = detect.getMD5(path_img)
        ch = bnwe_md5[0]
        print(path_img)
        os.rename(path_img, path_dir_root + "/" + ch + "/" + bnwe_md5 + ext_img)
        os.rename(path_csv, path_dir_root + "/" + ch + "/" + bnwe_md5 + ".csv")


# put name information
def conv7():
    path_dir_root = "/Volumes/PortableHDD/face/rect_cv"
    hex = [str(format(i, 'x')) for i in range(0, 16)]
    print(hex)
    for ch in hex:
        if not os.path.isdir(path_dir_root + "/" + ch):
            os.mkdir(path_dir_root + "/" + ch)

    for path_csv in glob.glob(path_dir_root + "/**/*.csv", recursive=True):
        dict_data = detect.get_csv_data(path_csv)
        bn = os.path.basename(path_csv)
        bnwe = bn.rsplit(".", 1)[0]
        path_dir_img = os.path.dirname(path_csv)
        ext_img = ".png"
        if not os.path.isfile(path_dir_img + "/" + bnwe + ext_img):
            ext_img = ".jpg"
        dict_data["url_name"] = bnwe + ext_img
        detect.save_csv_data(path_csv, dict_data)
        path_img = path_dir_img + "/" + bnwe + ext_img
        assert os.path.isfile(path_img)
        bnwe_md5 = detect.getMD5(path_img)
        print(bn, bnwe_md5)

# remove duplicated image. If a image is in X remove it from Z
def conv8():
    path_dir_root = "/Volumes/PortableHDD/face/rect_cv"
    for path_csv in glob.glob(path_dir_root + "/**/*.csv", recursive=True):
        dict_data = detect.get_csv_data(path_csv)
        if not "url_name" in dict_data:
            continue
        url_name = dict_data["url_name"]
        if not os.path.isfile(path_dir_root + "/x/" + url_name):
            continue
        bnwe_md5 = os.path.basename(path_csv).split(".", 1)[0]
        path_dir = path_csv.rsplit("/", 1)[0]
        ext = ".png"
        if not os.path.isfile(path_dir + "/" + bnwe_md5 + ext):
            ext = ".jpg"
        path_img = path_dir + "/" + bnwe_md5 + ext
        print(path_img)
        path_img_rect = path_dir + "/" + bnwe_md5 + "_rect.png"
        assert os.path.isfile(path_img_rect)
        assert os.path.isfile(path_csv)
        assert os.path.isfile(path_img)
        print(path_csv)
        print(path_img_rect)
        print(path_img)
        os.remove(path_csv)
        os.remove(path_img_rect)
        os.remove(path_img)

# look at Z folder, update the face rects
def conv9():
    detect_cv = detect.FaceDetectorCV()
    detect_dnn = detect.FaceDetectorDNN(".")
    path_dir_root = "/home/nobuyuki/project/face/rect_manual"
    for path_csv in glob.glob(path_dir_root + "/**/*.csv", recursive=True):
        dict_data = detect.get_csv_data(path_csv)
        bnwe_md5 = os.path.basename(path_csv).split(".", 1)[0]
        path_dir = path_csv.rsplit("/", 1)[0]
        ext = ".png"
        if not os.path.isfile(path_dir + "/" + bnwe_md5 + ext):
            ext = ".jpg"
        path_img = path_dir + "/" + bnwe_md5 + ext
#        print(path_img)
        img = cv2.imread(path_img)
        #### reject image if you don't like
        rect_cv2 = detect_cv.get_one_face(img)
        rect_dnn = detect_dnn.get_one_face(img)
#        print(rect_dnn)
        ####
        if rect_cv2 is not None:
            dict_data["rect_face_cv"] = str(rect_cv2[0]) + "," + str(rect_cv2[1]) + "," + str(rect_cv2[2]) + "," + str(
                rect_cv2[3])
        if rect_dnn is not None:
            dict_data["rect_face_dnn"] = str(rect_dnn[0]) + "," + str(rect_dnn[1]) + "," + str(rect_dnn[2]) + "," + str(
                rect_dnn[3])
        detect.save_csv_data(path_csv,dict_data)

# look at y folder, if it has multiple face move it to Z folder
def conv10():
    detect_cv = detect.FaceDetectorCV()
    detect_dnn = detect.FaceDetectorDNN('.')
    #path_dir_root = "/home/nobuyuki/project/face/umet55"
    #path_dir_root = "/media/nobuyuki/PortableHDD/umet55"
    path_dir_root = "/Volumes/PortableHDD/umet55"
    if os.uname()[1] == "nobuyuki-ThinkPad-T480":
        path_dir_root = "/media/nobuyuki/PortableHDD/umet55"

    for path_img in glob.glob(path_dir_root+"/xinbox/*"):
        img = cv2.imread(path_img)
        cv2.imshow('img', img)
        cv2.waitKey(50)
        bn = os.path.basename(path_img)

        #### reject image if you don't like
        ary_rect_cv2 = detect_cv.get_face(img)
        ary_rect_dnn = detect_dnn.get_face(img)
        print(ary_rect_dnn)
        if len(ary_rect_dnn) > 1: # many faces move to x
            print("move to trash: ",path_img)
            shutil.move(path_img, path_dir_root+"/xtrash/"+bn)
            continue

        dict_data = {}
        dict_data["url_name"] = bn
        if len(ary_rect_cv2) == 1:
            rect_cv2 = ary_rect_cv2[0]
            dict_data["rect_face_cv"] = str(rect_cv2[0])+","+str(rect_cv2[1])+","+str(rect_cv2[2])+","+str(rect_cv2[3])
        if len(ary_rect_dnn) == 1:
            rect_dnn = ary_rect_dnn[0]
            dict_data["rect_face_dnn"] = str(rect_dnn[0])+","+str(rect_dnn[1])+","+str(rect_dnn[2])+","+str(rect_dnn[3])
        dict_data["shape_img"] = str(img.shape[0]) + "," +str(img.shape[1])
        detect.draw_annotation(img, dict_data)

        cv2.imshow('img', img)
        cv2.waitKey(50)

        shutil.move(path_img, path_dir_root+"/xdifficult/"+bn)


        '''
        ### md5
        bnwe_md5 = detect.getMD5(path_img)
        ext = path_img.rsplit(".",1)[1]
        dir_dist = path_dir_root+"/"+str(bnwe_md5[0])
        path_img_new = dir_dist+"/"+bnwe_md5+"."+ext

        ### register image
        print("found new iamge:",path_img,path_img_new)

        if os.path.isfile(path_img_new):
            print("already exists",path_img_new)
            continue

        shutil.move(path_img,path_img_new)

        dict_data = {}
        dict_data["url_name"] = bn
        if len(ary_rect_cv2) == 1:
            rect_cv2 = ary_rect_cv2[0]
            dict_data["rect_face_cv"] = str(rect_cv2[0])+","+str(rect_cv2[1])+","+str(rect_cv2[2])+","+str(rect_cv2[3])
        if len(ary_rect_dnn) == 1:
            rect_dnn = ary_rect_dnn[0]
            dict_data["rect_face_dnn"] = str(rect_dnn[0])+","+str(rect_dnn[1])+","+str(rect_dnn[2])+","+str(rect_dnn[3])
        dict_data["shape_img"] = str(img.shape[0]) + "," +str(img.shape[1])
        detect.save_csv_data(dir_dist+"/"+bnwe_md5+".csv",dict_data)
        '''


#        size = (300, int(float(img.shape[0]) / img.shape[1] * 300.0))
#        img3 = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC);
#        compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
#        cv2.imwrite(dir_dist+"/"+bnwe_md5+"_anno.png", img3, compression_params)




# look at Z folder, put the sie information
def conv11():
    #path_dir_root = "/home/nobuyuki/project/face/rect_manual"
    #path_dir_root = "/Volumes/PortableHDD/rect_manual"
    path_dir_root = "/Volumes/PortableHDD/umet55"
    for path_csv in glob.glob(path_dir_root + "/**/*.csv", recursive=True):
        dict_data = detect.get_csv_data(path_csv)
        bnwe_md5 = os.path.basename(path_csv).split(".", 1)[0]
        path_dir = path_csv.rsplit("/", 1)[0]
        ext = ".png"
        if not os.path.isfile(path_dir + "/" + bnwe_md5 + ext):
            ext = ".jpg"
        path_img = path_dir + "/" + bnwe_md5 + ext
        img = cv2.imread(path_img)
        print(path_csv,img.shape)
        dict_data["shape_img"] = str(img.shape[0]) + "," +str(img.shape[1])
        detect.save_csv_data(path_csv,dict_data)

def conv12(path_dir_root):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        print("using GPU")
    else:
        print("using CPU")

    net_d,net_c,net_l = detect.load_detection_network(is_cuda,'.')
    ######
    print("select good and bad:", path_dir_root)
    list_path_img = []
    list_path_img = glob.glob(path_dir_root + '/xinbox/*.jpg', recursive=True) + list_path_img
    list_path_img = glob.glob(path_dir_root + '/xinbox/*.png', recursive=True) + list_path_img
    print("there is ",len(list_path_img),"images")
    for path_img in list_path_img:
        if not os.path.isfile(path_img):
            continue
        npImg = cv2.imread(path_img)
        if npImg is None:
            continue
        ####
        print(npImg.shape[0],npImg.shape[1])
        list_rect_dnn0 = detect.detect_face_dnn_multires(net_d, net_c, net_l, npImg, threshold_prob=0.6)
        for rect in list_rect_dnn0:
            detect.highlight_rect(npImg, rect, (255, 255, 0), width=1)
        list_rect_dnn = detect.marge_rects(list_rect_dnn0)
        for rect in list_rect_dnn:
            detect.highlight_rect(npImg, rect, (0, 255, 255), width=2)
        cv2.imshow('img',npImg)
        key = cv2.waitKey(100)
        ikey = int(key)
        print(ikey)
        if ikey == 113:
            exit()
        '''
        if key == 8: # delete key
            path_img_new = path_dir_root + '/xtrash/' + os.path.basename(path_img)
            print("move_to_trash")
            shutil.move(path_img, path_img_new)
            continue

        if ikey != 32: #space key
            print("this image is bad. do nothing")
            continue
        '''
        ### md5
        bnwe_md5 = detect.getMD5(path_img)
        ext = path_img.rsplit(".",1)[1]
        dir_dist = path_dir_root+"/"+str(bnwe_md5[0])
        path_img_new = dir_dist+"/"+bnwe_md5+"."+ext
        print(path_img_new,dir_dist)

        if os.path.isfile(path_img_new):
            print("there is already one:")
            os.remove(path_img)
            continue

        ### register image
        print("found new iamge:",path_img,path_img_new)

        shutil.move(path_img,path_img_new)

        dict_data = {}
        dict_data["url_name"] = os.path.basename(path_img)
        dict_data["shape_img"] = str(npImg.shape[0]) + "," +str(npImg.shape[1])
        if len(list_rect_dnn) > 0:
            str_csv = detect.get_str_csv_list_rect(list_rect_dnn)
            dict_data["rect_face_dnn"] = str_csv
        detect.save_csv_data(dir_dist+"/"+bnwe_md5+".csv",dict_data)


if __name__ == "__main__":
    #	conv1()
    #	conv2()
    #	conv3()
    #   conv4()
    #    conv5()
    #    conv6()
    #    conv7()
    #conv9()
    #conv10()
    conv12("/media/nobuyuki/PortableHDD/face/umet55")

