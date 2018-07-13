
import cv2
import os, numpy, math, hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable


class FaceDetectorCV:
    def __init__(self):
        path_data = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(path_data + 'haarcascade_frontalface_default.xml')

    def get_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, minNeighbors=10)
        return faces


class FaceDetectorDNN:
    def __init__(self, dir_path, threshold_prob=0.8):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            print("using GPU")
        else:
            print("using CPU")
        self.net_d, self.net_c, self.net_l = load_detection_network(is_cuda, dir_path)
        self.threshold_prob = threshold_prob

    def get_face(self, img):
        aRect = detect_face_dnn_multires(self.net_d, self.net_c, self.net_l, img, self.threshold_prob)
        aRect = marge_rects(aRect)
        return aRect


def draw_annotation(img, dict_data):
    if "rect_face_cv" in dict_data:
        rect_cv = list(map(float, dict_data["rect_face_cv"].split(",")))
        if "no_face" in dict_data or "rect_face_dnn" in dict_data or "rect_face_manual" in dict_data:
            highlight_rect(img, rect_cv, (255, 0, 0), width=1)
        else:
            highlight_rect(img, rect_cv, (255, 0, 0), width=2)
    if "rect_face_dnn" in dict_data:
        rect_dnn = list(map(float, dict_data["rect_face_dnn"].split(",")))
        if "no_face" in dict_data or "rect_face_manual" in dict_data:
            highlight_rect(img, rect_dnn, (0, 255, 0), width=1)  # green
        else:
            highlight_rect(img, rect_dnn, (0, 255, 0), width=2)  # green
    if "rect_face_manual" in dict_data:
        rect_manual = list(map(float, dict_data["rect_face_manual"].split(",")))
        if "no_face" in "dict_data":
            highlight_rect(img, rect_manual, (0, 0, 255), width=1)
        else:
            highlight_rect(img, rect_manual, (0, 0, 255), width=2)


def detect_face_cv(img, face_cascade, eye_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, minNeighbors=10)
    face_eye = None
    for [x, y, w, h] in faces:
        face_eye0 = [[x, y, w, h]]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = img[y:y + h, x:x + w]
        face_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_gray, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            face_eye0.append([ex, ey, ew, eh])
        if face_eye is None:
            face_eye = [face_eye0]
        else:
            face_eye.append(face_eye0)
    if face_eye is None:
        face_eye = [[]]
    return face_eye


def get_imgage_train(path_csv):
    assert os.path.isfile(path_csv)
    path_input = path_csv.rsplit('/', 1)[0]
    bn = os.path.basename(path_csv)
    bn, ext = bn.rsplit(".", 1)[:2]
    assert ext == "csv"
    #	print(path,path_input,bn)
    path_img = path_input + '/' + bn + '.png'
    if not os.path.isfile(path_img):
        path_img = path_input + '/' + bn + '.jpg'
    if not os.path.isfile(path_img):
        print(path_img)
        assert False
    img1 = cv2.imread(path_img)
    return img1


def get_csv_data(path_csv):
    assert os.path.isfile(path_csv)
    path_input = path_csv.rsplit('/', 1)[0]
    bn = os.path.basename(path_csv)
    assert bn.rsplit(".", 1)[1] == "csv"  # check extension
    lines = []
    with open(path_csv, 'r') as fin:
        lines = fin.readlines()
    #    print(lines)
    dict_data = {}
    for line in lines:
        aS = line.split(',', 1)
        if len(aS) != 2: continue
        name = aS[0]
        data = aS[1]
        if data.endswith("\n"):
            data = data.split("\n")[0]
        #        print(name,data)
        dict_data[name] = data
    return dict_data


def getMD5(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_str_csv_list_rect(list_rect):
    assert (len(list_rect) > 0)
    str_csv = ""
    for il, rect in enumerate(list_rect):
        str_csv += str(rect[0]) + "," + str(rect[1]) + "," + str(rect[2]) + "," + str(rect[3])
        if il != len(list_rect) - 1:
            str_csv += ","
    return str_csv


def get_list_rect_from_string(rect_str):
    list_rect = list(map(float, rect_str.split(",")))
    assert len(list_rect) % 4 == 0
    list_rect = [list_rect[i:i + 4] for i in range(0, len(list_rect), 4)]
    return list_rect


def save_csv_data(path_csv, dict_data) -> None:
    with open(path_csv, "w") as fout:
        for key in dict_data:
            val = dict_data[key]
            fout.write(key + "," + val + "\n")


def get_list_rect_from_dict_info(dict_data: dict) -> list:
    if "no_face" in dict_data:
        return []
    elif "rect_face_manual" in dict_data:
        rect_str = dict_data["rect_face_manual"]
    elif "rect_face_dnn" in dict_data:
        rect_str = dict_data["rect_face_dnn"]
    elif "rect_face_cv" in dict_data:
        rect_str = dict_data["rect_face_cv"]
    else:
        return []
    return get_list_rect_from_string(rect_str)


def swap_keyword(key_old, key_new, dict0):
    if key_old in dict0:
        val = dict0[key_old]
        del dict0[key_old]
        dict0[key_new] = val


'''
def get_image_rect_train(path_csv, npix, nblock):
    img1 = get_imgage_train(path_csv)
    ####
    dict_data = get_csv_data(path_csv)
    rect = get_rect_train(dict_data)
    ####
    scale_inv = 1      
    if rect is not None:
        assert len(rect) == 4
        if 64 < rect[2] and rect[2] < 128:
            scale_inv = random.randint(2,4)
        if 128 < rect[2] and rect[2] < 256:
            scale_inv = random.randint(4,8)
        if 256 < rect[2] and rect[2] < 512:
            scale_inv = random.randint(8,16)
        if 512 < rect[2] and rect[2] < 1024:
            scale_inv = random.randint(16,32)
    else: # image fit into the square
        scale_inv3 = math.ceil( max(img1.shape[0],img1.shape[1])/(npix*nblock))
        scale_inv4 = math.ceil(min(img1.shape[0],img1.shape[1])/(npix*nblock))        
        scale_inv = random.randint(scale_inv4,scale_inv3)
    if random.uniform(0,1) < 0.2:  # removing the bias that small image have big face
        scale_inv1 = math.ceil(min(img1.shape[0],img1.shape[1])/(npix*1))
        scale_inv2 = math.ceil(min(img1.shape[0],img1.shape[1])/(npix*2))      
        scale_inv = random.randint(scale_inv2,scale_inv1)
#        print(scale_inv,scale_inv2,scale_inv1)
    scale = 1.0/float(scale_inv)
    ####
    trans_hw2 = [0.0, 0.0]
    rect2 = None
    if rect is not None:
        rect2 = [x*scale for x in rect]
        if (rect2[0]+rect2[2]*0.5) > npix*(nblock-1): # width exceeding limit
            trans_hw2[1] = rect2[0]+rect2[2]*0.5 - npix*nblock*0.5
        if (rect2[1]+rect2[3]*0.5) > npix*(nblock-1): # height exceeding limit
            trans_hw2[0] = rect2[1]+rect2[3]*0.5 - npix*nblock*0.5
        trans_hw2[0] += random.randint(0,math.ceil(32.0/scale_inv))
        trans_hw2[1] += random.randint(0,math.ceil(32.0/scale_inv))
        rect2[0] -= trans_hw2[1]
        rect2[1] -= trans_hw2[0]
    img2 = img1[int(trans_hw2[0]*scale_inv)::scale_inv,int(trans_hw2[1]*scale_inv)::scale_inv]
    # print(img1.shape,img2.shape,scale_inv,trans_hw2)
    ####
    img3 = img2.copy()
    if img3.shape[0] < npix * nblock:
        img3 = cv2.copyMakeBorder(img3, 0, npix * nblock -img3.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if img3.shape[1] < npix * nblock:
        img3 = cv2.copyMakeBorder(img3, 0, 0, 0, npix * nblock - img3.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img3 = img3[:npix*nblock,:npix*nblock]
    ####
    tnsr = numpy.moveaxis(img3, 2, 0).astype(numpy.float32) / 255.0
    ####
    return img3, rect2, tnsr
'''


def get_image_rect_train(path_csv, npix, mag, transh, transw):
    img1 = get_imgage_train(path_csv)
    list_rect = get_list_rect_from_dict_info(get_csv_data(path_csv))
    ####
    list_rect2 = []
    for rect in list_rect:
        rect2 = [float(x) / mag for x in rect]
        list_rect2.append([rect2[0] - transw, rect2[1] - transh, rect2[2], rect2[3]])

    img2 = img1[transh * mag::mag, transw * mag::mag]
    ####
    img3 = img2.copy()
    h3 = img3.shape[0]
    w3 = img3.shape[1]
    nbh3 = math.ceil(h3 / npix)
    nbw3 = math.ceil(w3 / npix)
    img3 = cv2.copyMakeBorder(img3, 0, npix * nbh3 - h3, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img3 = cv2.copyMakeBorder(img3, 0, 0, 0, npix * nbw3 - w3, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    ####
    tnsr = numpy.moveaxis(img3, 2, 0).astype(numpy.float32) / 255.0
    ####
    return img3, list_rect2, tnsr


# Get N image and rectanble from aPath
def get_batch(list_path_mag, npix, nblock, threshold_iou, ratio_margin_iou):
    np_batch_in = []
    np_batch_trg_c = []
    np_batch_trg_l = []
    np_batch_mask = []
    nelem = len(list_path_mag)
    for i in range(nelem):
        assert len(list_path_mag[i]) == 4
        path = list_path_mag[i][0]
        mag = list_path_mag[i][1]
        transh = list_path_mag[i][2]
        transw = list_path_mag[i][3]
        img, list_rect, tnsr = get_image_rect_train(path, npix, mag, transh, transw)
        assert img.shape[0] == nblock[0] * npix
        assert img.shape[1] == nblock[1] * npix
        ol = overlap_rect(list_rect, npix, img.shape, threshold_iou, ratio_margin_iou)
        assert ol.shape[0] == nblock[0] and ol.shape[1] == nblock[1]
        ###
        trgl = numpy.zeros((3, nblock[0], nblock[1]))
        mask = numpy.zeros((nblock[0], nblock[1]))
        for ih in range(nblock[0]):
            for iw in range(nblock[1]):
                if ol[ih, iw] != 1:
                    continue
                rect0 = [npix * iw, npix * ih, npix, npix]
                list_iou_ir = []
                for ir, rect in enumerate(list_rect):
                    iou = iou_rect(rect, rect0)
                    list_iou_ir.append((iou, ir))
                list_iou_ir = sorted(list_iou_ir, reverse=True)
                rect = list_rect[list_iou_ir[0][1]]
                trgl[0, ih, iw] = float(rect[0] + rect[2] * 0.5) / npix - (iw + 0.5)
                trgl[1, ih, iw] = float(rect[1] + rect[2] * 0.5) / npix - (ih + 0.5)
                trgl[2, ih, iw] = math.log(float(rect[2]) / npix, 2) * 0.5
                mask[ih, iw] = 1.0
        ###
        np_batch_in.append(tnsr.flatten())
        np_batch_trg_c.append(ol.flatten().astype(numpy.int64))
        np_batch_trg_l.append(trgl.flatten().astype(numpy.float32))
        np_batch_mask.append(mask.flatten().astype(numpy.float32))
    np_batch_in = numpy.asarray(np_batch_in).reshape(nelem, 3, npix * nblock[0], npix * nblock[1])
    np_batch_trg_c = numpy.asarray(np_batch_trg_c).reshape(nelem, nblock[0], nblock[1])
    np_batch_trg_l = numpy.asarray(np_batch_trg_l).reshape(nelem, 3, nblock[0], nblock[1])
    np_batch_mask = numpy.asarray(np_batch_mask).reshape(nelem, nblock[0], nblock[1])
    return np_batch_in, np_batch_trg_c, np_batch_trg_l, np_batch_mask


def intersect_rects(rec0, rec1):
    assert len(rec0) == 4
    assert len(rec1) == 4
    tlx = max(rec0[0], rec1[0])
    tly = max(rec0[1], rec1[1])
    brx = min(rec0[0] + rec0[2], rec1[0] + rec1[2])
    bry = min(rec0[1] + rec0[3], rec1[1] + rec1[3])
    if tlx > brx: return []
    if tly > bry: return []
    return [tlx, tly, brx - tlx, bry - tly]


def area_rect(rec0):
    if len(rec0) != 4: return 0.0
    return rec0[2] * rec0[3]


def iou_rect(trim0, trim2):
    trim3 = intersect_rects(trim0, trim2)
    an = area_rect(trim3)
    au = area_rect(trim2) + area_rect(trim0) - an
    iou = float(an) / float(au)
    return iou
    
    

def overlap_rect(list_rect, npix, shape_img, threshold_iou, ratio_margin_iou):
    nblock_h = int(math.ceil(shape_img[0] / npix))
    nblock_w = int(math.ceil(shape_img[1] / npix))
    ol = numpy.zeros((nblock_h, nblock_w))
    for ih in range(nblock_h):
        for iw in range(nblock_w):
            rec0 = [iw * npix, ih * npix, npix, npix]
            for rect in list_rect:
                iou = iou_rect(rec0, rect)
                if iou < threshold_iou * ratio_margin_iou:
                    pass
                elif iou < threshold_iou and ol[ih, iw] == 0:
                    ol[ih, iw] = -1
                else:
                    ol[ih, iw] = 1
    return ol


def highlight_rect(img, rect, color, width=2):
    assert len(rect) == 4
    recti = list(map(int, rect))
    cv2.rectangle(img, (recti[0], recti[1]), (recti[0] + recti[2], recti[1] + recti[3]), color, width)


def scale_rect(list_rect, s):
    for rect in list_rect:
        assert len(rect) == 4
        rect[0] = s * rect[0]
        rect[1] = s * rect[1]
        rect[2] = s * rect[2]
        rect[3] = s * rect[3]
    return list_rect


def marge_rects(aRect):
    nl = len(aRect)
    if nl == 0: return []
    #	print(nl)
    A = numpy.zeros((nl, nl))
    for il in range(nl):
        for jl in range(il, nl):
            rect_i = aRect[il]
            rect_j = aRect[jl]
            iou = iou_rect(rect_i, rect_j)
            #			print(rect_i,rect_j,iou)
            A[il, jl] = iou
            A[jl, il] = iou

    w, v = numpy.linalg.eig(A)
    ind = w.argsort()[::-1]
    vt = v.transpose()
    #    vt = v

    #	print("######")
    #	print(A)
    #	print(w)
    #	print(v)
    #	print(ind)

    aRectC = []
    for jl in range(nl):
        i0 = ind[jl]
        w0 = w[i0]
        if w0 < 0.9:
            break
        ###
        vt0 = vt[i0]
        #		print(i0,w0,vt0)
        R0 = R1 = R2 = 0
        W = 0
        for il in range(nl):
            W += vt0[il]
        if abs(W) < 1.0e-5:
            continue
        isValid = True
        for il in range(nl):
            if vt0[il] / W < -0.1:
                isValid = False
                break
            R0 += (aRect[il][0] + aRect[il][2] * 0.5) * vt0[il]
            R1 += (aRect[il][1] + aRect[il][2] * 0.5) * vt0[il]
            R2 += aRect[il][2] * vt0[il]
        if not isValid:
            continue
        R0 /= W
        R1 /= W
        R2 /= W
        aRectC.append([R0 - R2 * 0.5, R1 - R2 * 0.5, R2, R2])

    return aRectC


def normalizeTrchTnsrImg(tnsrTImg):
    mean0 = tnsrTImg[0, :, :].mean()
    mean1 = tnsrTImg[1, :, :].mean()
    mean2 = tnsrTImg[2, :, :].mean()
    stdev0 = tnsrTImg[0, :, :].std()
    stdev1 = tnsrTImg[1, :, :].std()
    stdev2 = tnsrTImg[2, :, :].std()
    stdev0 = 1.0 / stdev0
    stdev1 = 1.0 / stdev1
    stdev2 = 1.0 / stdev2
    tnsrTImg[0, :, :] -= mean0
    tnsrTImg[1, :, :] -= mean1
    tnsrTImg[2, :, :] -= mean2
    tnsrTImg[0, :, :] *= stdev0
    tnsrTImg[1, :, :] *= stdev1
    tnsrTImg[2, :, :] *= stdev2


###################################################################


def load_detection_network(is_cuda, path_dir):
    ####
    net_d = NetDesc()
    if os.path.isfile(path_dir + '/model_d'):
        if is_cuda:
            net_d.load_state_dict(torch.load(path_dir + '/model_d'))
        else:
            net_d.load_state_dict(torch.load(path_dir + '/model_d', map_location='cpu'))

    ####
    net_c = NetClass()
    if os.path.isfile(path_dir + '/model_c'):
        if is_cuda:
            net_c.load_state_dict(torch.load(path_dir + '/model_c'))
        else:
            net_c.load_state_dict(torch.load(path_dir + '/model_c', map_location='cpu'))

    ####
    net_l = NetLoc()
    if os.path.isfile(path_dir + '/model_l'):
        if is_cuda:
            net_l.load_state_dict(torch.load(path_dir + '/model_l'))
        else:
            net_l.load_state_dict(torch.load(path_dir + '/model_l', map_location='cpu'))

    if is_cuda:
        net_d = net_d.cuda()
        net_c = net_c.cuda()
        net_l = net_l.cuda()

    return net_d, net_c, net_l


def pad32TrchV(trchVImg0):
    npix = 32
    h0 = trchVImg0.size()[1]
    w0 = trchVImg0.size()[2]
    H0 = math.ceil(h0 / npix)
    W0 = math.ceil(w0 / npix)
    trchV = F.pad(trchVImg0, (0, W0 * npix - w0, 0, H0 * npix - h0))
    return trchV


def detect_face_dnn(net_d, net_c, net_l, trchVImg, threshold_prob):
    ###
    npix = net_d.npix
    is_cuda = next(net_d.parameters()).is_cuda
    assert next(net_c.parameters()).is_cuda == is_cuda
    assert next(net_l.parameters()).is_cuda == is_cuda
    sm = nn.Softmax(dim=1)
    ###
    trchVImg2 = trchVImg.view(1, 3, trchVImg.shape[1], trchVImg.shape[2])
    #	print(input_img2.shape,input_img.shape)
    descriptor = net_d(trchVImg2)
    nblockH = descriptor.shape[2]
    nblockW = descriptor.shape[3]
    assert trchVImg.shape[1] == nblockH * npix
    assert trchVImg.shape[2] == nblockW * npix
    #	print(descriptor.shape,nblockH,nblockW)
    output_c = net_c(descriptor)
    output_c = sm(output_c)
    output_l = net_l(descriptor)
    if is_cuda:
        np_desc = descriptor.cpu().data.numpy()
        np_outc = output_c.cpu().data.numpy()
        np_outl = output_l.cpu().data.numpy()
    else:
        np_desc = descriptor.data.numpy()
        np_outc = output_c.data.numpy()
        np_outl = output_l.data.numpy()
    #	print(descriptor.shape, output_c.shape, output_l.shape)

    aRect = []
    for ih in range(nblockH):
        for iw in range(nblockW):
            rect0 = [npix * iw, npix * ih, npix, npix]
            if np_outc[0, 1, ih, iw] < threshold_prob: continue
            dx = np_outl[0, 0, ih, iw]
            dy = np_outl[0, 1, ih, iw]
            ds = np_outl[0, 2, ih, iw]
            r2 = math.pow(2, ds * 2) * npix
            r0 = (dx + iw + 0.5) * npix - r2 * 0.5
            r1 = (dy + ih + 0.5) * npix - r2 * 0.5
            #            print(dx,dy,ds,"  ",r0,r1,r2, math.pow(2,ds*2),npix)
            rect1 = [r0, r1, r2, r2]
            aRect.append(rect1)
    return aRect


def detect_face_dnn_multires(net_d, net_c, net_l, npImg, threshold_prob, nblk_max=15):
    is_cuda = next(net_d.parameters()).is_cuda
    assert next(net_c.parameters()).is_cuda == is_cuda
    assert next(net_l.parameters()).is_cuda == is_cuda
    net_d.eval()
    net_c.eval()
    net_l.eval()
    convNp2Tensor = torchvision.transforms.ToTensor()
    trchVImg0 = Variable(convNp2Tensor(npImg), requires_grad=False)
    if is_cuda:
        trchVImg0 = trchVImg0.cuda()

    ####
    list_rect = []
    #    trchVImg0 = trchVImg0[:, ::1, ::1]
    trchVImg0a = pad32TrchV(trchVImg0)
    if max(trchVImg0a.shape[1:3]) / 32 < nblk_max and min(trchVImg0a.shape[1:3]) / 32 > 2:
        list_rect0 = detect_face_dnn(net_d, net_c, net_l, trchVImg0a, threshold_prob)
        list_rect = list_rect + scale_rect(list_rect0, 1.0)
    ####
    trchVImg1 = trchVImg0[:, ::2, ::2]
    trchVImg1a = pad32TrchV(trchVImg1)
    if max(trchVImg1a.shape[1:3]) / 32 < nblk_max and min(trchVImg1a.shape[1:3]) / 32 > 2:
        list_rect1 = detect_face_dnn(net_d, net_c, net_l, trchVImg1a, threshold_prob)
        list_rect = list_rect + scale_rect(list_rect1, 2.0)
    ####
    trchVImg2 = trchVImg0[:, ::4, ::4]
    trchVImg2a = pad32TrchV(trchVImg2)
    if max(trchVImg2a.shape[1:3]) / 32 < nblk_max and min(trchVImg2a.shape[1:3]) / 32 > 2:
        list_rect2 = detect_face_dnn(net_d, net_c, net_l, trchVImg2a, threshold_prob)
        list_rect = list_rect + scale_rect(list_rect2, 4.0)
    ####
    trchVImg3 = trchVImg0[:, ::8, ::8]
    trchVImg3a = pad32TrchV(trchVImg3)
    if max(trchVImg3a.shape[1:3]) / 32 < nblk_max and min(trchVImg3a.shape[1:3]) / 32 > 2:
        list_rect3 = detect_face_dnn(net_d, net_c, net_l, trchVImg3a, threshold_prob)
        list_rect = list_rect + scale_rect(list_rect3, 8.0)
    return list_rect


def detect_face_dnn_multires_merge(net_d, net_c, net_l, npImg):
    list_rect = detect_face_dnn_multires(net_d, net_c, net_l, npImg)
    list_rect_merge = marge_rects(list_rect)
    return list_rect_merge


def initialize_net(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight, 1.414)
            nn.init.constant(m.bias, 0.1)
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class NetClass(nn.Module):
    def __init__(self):
        super(NetClass, self).__init__()
        self.nchanel_in = 256
        self.conv1 = nn.Conv2d(256, 64, kernel_size=1)
        #		self.bn1   = nn.BatchNorm2d(64)
        ###
        self.conv2 = nn.Conv2d(64, 16, kernel_size=1)
        #		self.bn2    = nn.BatchNorm2d(16)
        ####
        self.conv3 = nn.Conv2d(16, 2, kernel_size=1)
        #		self.bn3    = nn.BatchNorm2d(2)
        ###
        initialize_net(self)

    def forward(self, x):
        #		x = F.relu(self.bn1(self.conv1(x)))
        #		x = F.relu(self.bn2(self.conv2(x)))
        #		x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class NetLoc(nn.Module):
    def __init__(self):
        super(NetLoc, self).__init__()
        self.nchanel_in = 256
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1)
        #		self.bn1   = nn.BatchNorm2d(128)
        ####
        self.conv2 = nn.Conv2d(128, 32, kernel_size=1)
        #		self.bn2   = nn.BatchNorm2d(32)
        ####
        self.conv3 = nn.Conv2d(32, 16, kernel_size=1)
        #		self.bn3   = nn.BatchNorm2d(16)
        ####
        self.conv4 = nn.Conv2d(16, 3, kernel_size=1)
        #		self.bn4   = nn.BatchNorm2d(3)
        ####
        self.af = nn.Tanh()
        initialize_net(self)

    def forward(self, x):
        #		x = self.af(self.bn1(self.conv1(x)))
        #		x = self.af(self.bn2(self.conv2(x)))
        #		x = self.af(self.bn3(self.conv3(x)))
        #		x = self.af(self.bn4(self.conv4(x)))
        x = self.af(self.conv1(x))
        x = self.af(self.conv2(x))
        x = self.af(self.conv3(x))
        x = self.af(self.conv4(x))
        return x


class NetUnit(nn.Module):
    def __init__(self, nc):
        super(NetUnit, self).__init__()
        self.bn1 = nn.BatchNorm2d(nc)
        self.conv1 = nn.Conv2d(nc, nc, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(nc)
        self.conv2 = nn.Conv2d(nc, nc, kernel_size=3, padding=1)
        initialize_net(self)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + x


class NetDesc(nn.Module):
    def __init__(self):
        super(NetDesc, self).__init__()
        ####
        self.conv1a = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)  # 1/2
        ####
        self.unit2a = NetUnit(64)
        self.unit2b = NetUnit(64)
        self.unit2c = NetUnit(64)
        self.bn2 = nn.BatchNorm2d(64)
        ####
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)  # 1/2
        self.unit3a = NetUnit(128)
        self.unit3b = NetUnit(128)
        self.unit3c = NetUnit(128)
        self.unit3d = NetUnit(128)
        self.bn3 = nn.BatchNorm2d(128)
        ####
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)  # 1/2
        self.unit4a = NetUnit(256)
        self.unit4b = NetUnit(256)
        self.unit4c = NetUnit(256)
        self.unit4d = NetUnit(256)
        self.bn4 = nn.BatchNorm2d(256)
        ####
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)  # 1/2
        self.unit5a = NetUnit(256)
        self.unit5b = NetUnit(256)
        self.unit5c = NetUnit(256)
        self.bn5 = nn.BatchNorm2d(256)
        ####
        self.npix = 32
        self.nchanel_out = 256
        ####
        initialize_net(self)

    def forward(self, x):
        x = self.conv1a(x)
        ####
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # 1/2
        x = self.unit2a(x)
        x = self.unit2b(x)
        x = self.unit2c(x)
        x = F.relu(self.bn2(x))
        #####
        x = self.conv3(x)
        x = self.unit3a(x)
        x = self.unit3b(x)
        x = self.unit3c(x)
        x = self.unit3d(x)
        x = F.relu(self.bn3(x))
        #####
        x = self.conv4(x)
        x = self.unit4a(x)
        x = self.unit4b(x)
        x = self.unit4c(x)
        x = self.unit4d(x)
        x = F.relu(self.bn4(x))
        #####
        x = self.conv5(x)
        x = self.unit5a(x)
        x = self.unit5b(x)
        x = self.unit5c(x)
        x = F.relu(self.bn5(x))
        return x
