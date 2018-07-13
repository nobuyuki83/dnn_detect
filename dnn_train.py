import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import cv2

import glob, os, numpy
import random, math
import pickle

import detect


def size_block(size, mag):
    nbh2 = math.ceil(size[0] / mag)
    nbw2 = math.ceil(size[1] / mag)
    h = math.ceil(nbh2 / 32)
    w = math.ceil(nbw2 / 32)
    return (h, w)

def make_dict_nblk_path(list_path_csv,min_nblk=2,max_nblk=16):
    dict_nblk_path = {}
    for path_csv in list_path_csv:
        dict_data = face.get_csv_data(path_csv)
        size_img = [0, 0]
        if "shape_img" in dict_data:
            size_img[0] = int(dict_data["shape_img"].split(",")[0])
            size_img[1] = int(dict_data["shape_img"].split(",")[1])
        else:
            img = face.get_imgage_train(path_csv)
            size_img = img.shape[:2]

        list_rect = face.get_list_rect_from_dict_info(dict_data)

        list_mag = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        list_trans = [(0, 0), (16, 0), (0, 16), (16, 16)]  # (h,w)
        for mag in list_mag:
            for trans in list_trans:
                is_crop = False
                for rect in list_rect:
                    if (rect is not None) and (rect[0] < trans[1] * mag or rect[1] < trans[0] * mag):
                        is_crop = True
                        break
                if is_crop:
                    continue
                size_img2 = (size_img[0] - trans[0] * mag, size_img[1] - trans[1] * mag)
                nb = size_block(size_img2, mag)
                if max(nb) <= max_nblk and min(nb) > min_nblk:
                    if not nb in dict_nblk_path:
                        dict_nblk_path[nb] = []
                    dict_nblk_path[nb].append((path_csv, mag, trans[0], trans[1]))
    return dict_nblk_path

class TrainingData:
    def __init__(self, list_path_dir, batch_size0):
        self.dict_nblock_path = {}
        for path_dir in list_path_dir:
            assert os.path.isdir(path_dir)
            if os.path.isfile(path_dir + "/dict_nblock_path.p"):
                dict0 = pickle.load(open(path_dir + "/dict_nblock_path.p", "rb"))
            else:
                list_path_csv = glob.glob(path_dir+"/**/*.csv")
                dict0 = make_dict_nblk_path(list_path_csv, min_nblk=2, max_nblk=16)
                pickle.dump(dict0, open(path_dir + "/dict_nblock_path.p", "wb"))
            ####
            for nb in dict0.keys():
                if not nb in self.dict_nblock_path:
                    self.dict_nblock_path[nb] = []
                self.dict_nblock_path[nb].extend(dict0[nb])

#        print(self.dict_nblock_path)

        # make random parmulation for each block size
        self.dict_permu = {}
        for nb in self.dict_nblock_path.keys():
            N = len(self.dict_nblock_path[nb])
            self.dict_permu[nb] = list(range(N))
            random.shuffle(self.dict_permu[nb])

        # make batches
        self.iepoch = 0
        self.batch_size = batch_size0
        self.list_batch = []
        self.list_blk = list(self.dict_nblock_path.keys())
        for iblk in range(len(self.list_blk)):
            blk = self.list_blk[iblk]
            npath = len(self.dict_nblock_path[blk])
            nbatch = int(math.ceil(float(npath) / self.batch_size))
            for iblkbatch in range(nbatch):
                self.list_batch.append((iblk, iblkbatch))
        random.shuffle(self.list_batch)
        self.ibatch = 0
        print("number_of_batch:",len(self.list_batch))

    def get_batch(self):
#        print(self.ibatch,len(self.list_batch))
        assert 0 <= self.ibatch < len(self.list_batch)
        iblk = self.list_batch[self.ibatch][0]
        iblkbatch = self.list_batch[self.ibatch][1]
        nb = self.list_blk[iblk]
        list_path_mag = self.dict_nblock_path[nb]
        list_parmu = self.dict_permu[nb]
        ####
        list_path_mag_batch = []
        for iielem in range(self.batch_size):
            ipath_mag = iblkbatch * self.batch_size + iielem
            if ipath_mag >= len(list_path_mag):
                break
            jpath_mag = list_parmu[ipath_mag]  # permutation
            list_path_mag_batch.append(list_path_mag[jpath_mag])

        self.ibatch = self.ibatch + 1
        if self.ibatch >= len(self.list_batch):
            self.ibatch = 0
            self.iepoch += 1
            random.shuffle(self.list_batch)
        return list_path_mag_batch, nb


# train descepter and classifier
def train(net_d, net_c, net_l, is_cuda, nitr, is_d, is_c, is_l,
          training_data,
          threshould_iou, ratio_marging_iou):
    assert isinstance(is_d, bool)
    assert isinstance(is_c, bool)
    assert isinstance(is_l, bool)
    assert net_l.nchanel_in == net_d.nchanel_out
    assert net_c.nchanel_in == net_d.nchanel_out

    difference_l = nn.SmoothL1Loss(reduce=False)
    difference_c = nn.CrossEntropyLoss(ignore_index=-1)

    param = []
    if is_d: param = list(net_d.parameters()) + param
    if is_l: param = list(net_l.parameters()) + param
    if is_c: param = list(net_c.parameters()) + param
    optimizer = optim.Adam(param, lr=0.0001)

    if is_d:
        net_d.train()
    else:
        net_d.eval()
    if is_l:
        net_l.train()
    else:
        net_l.eval()
    if is_c:
        net_c.train()
    else:
        net_c.eval()

    for itr in range(nitr):
        list_path_mag, nb = training_data.get_batch()
        # print(nb,list_path_mag)
        np_in, np_trgc, np_trgl, np_mask = face.get_batch(list_path_mag, 32, nb, threshould_iou, ratio_marging_iou)
        #		print(np_in.shape, np_trgc.shape,np_trgl.shape,np_mask.shape)
        input = Variable(torch.from_numpy(np_in), requires_grad=True)
        target_c = Variable(torch.from_numpy(np_trgc), requires_grad=False)
        target_l = Variable(torch.from_numpy(np_trgl), requires_grad=False)
        mask = Variable(torch.from_numpy(np_mask), requires_grad=False)
        if is_cuda:
            input = input.cuda()
            target_c = target_c.cuda()
            target_l = target_l.cuda()
            mask = mask.cuda()
        ###
        optimizer.zero_grad()
        descriptor = net_d(input)
        ###
        output_c = net_c(descriptor)
        loss_c = difference_c(output_c, target_c)
        ###
        output_l = net_l(descriptor)
        diffl = difference_l(output_l, target_l)
        loss_l = torch.dot(torch.sum(diffl, dim=1).view(-1), mask.view(-1, ))
        ###
        loss = loss_c + loss_l.mul(5.0e-4)
        ####
        print(training_data.iepoch, "/", training_data.ibatch, "/", len(training_data.list_batch), nb,
              " ", loss.data[0], loss_c.data[0], loss_l.data[0])
        #		print(itr," ",loss.data[0])
        loss.backward()
        optimizer.step()

    if is_d: torch.save(net_d.state_dict(), 'model_d.pt')
    if is_c: torch.save(net_c.state_dict(), 'model_c.pt')
    if is_l: torch.save(net_l.state_dict(), 'model_l.pt')


def view_train(training_data, threshould_iou, ratio_margnin_iou):
    npix = 32
    for itr in range(10000):
        list_path_mag, nb = training_data.get_batch()
        np_tnsr, np_trgc, np_trgl, np_mask = face.get_batch(list_path_mag, 32, nb, threshould_iou, ratio_margnin_iou)
        np_img = np_tnsr.reshape(3, nb[0] * 32, nb[1] * 32)
        np_img = numpy.moveaxis(np_img, 0, 2)
        np_img = (np_img * 255.0).astype(numpy.uint8)
        np_img = np_img.copy()
        ####
        path_csv = list_path_mag[0][0]
        print(training_data.iepoch, "/", training_data.ibatch, "/", len(training_data.list_batch),"  ", np_tnsr.shape,"  ",path_csv)
        list_rect = face.get_list_rect_from_dict_info(face.get_csv_data(path_csv))
        for rect in list_rect:
            mag = list_path_mag[0][1]
            rect = [x / mag for x in rect]
            rect[0] -= list_path_mag[0][3]
            rect[1] -= list_path_mag[0][2]
            face.highlight_rect(np_img, rect, (0, 0, 255))        
        ###
        for ih in range(nb[0]):
            for iw in range(nb[1]):
                rect1 = [npix * iw + 2, npix * ih + 2, npix, npix]
                if np_trgc[0, ih, iw] == 1:  # target
                    dx = np_trgl[0, 0, ih, iw]
                    dy = np_trgl[0, 1, ih, iw]
                    ds = np_trgl[0, 2, ih, iw]
                    r2 = math.pow(2, ds * 2) * npix
                    r0 = (dx + iw + 0.5) * npix - r2 * 0.5
                    r1 = (dy + ih + 0.5) * npix - r2 * 0.5
                    rect2 = [r0, r1, r2, r2]
                    face.highlight_rect(np_img, rect1, (255, 0, 255), width=1)                    
                    face.highlight_rect(np_img, rect2, (0, 255, 255), width=1)  # output rect
                    
                if np_trgc[0, ih, iw] == -1:  # target
                    face.highlight_rect(np_img, rect1, (255, 255, 0), width=1)
        ####
        cv2.imshow('img', np_img)
        ikey = int(cv2.waitKey(1000))
        if ikey == 32: 
            cv2.waitKey(5000)


def evaluate(net_d, net_c, net_l, is_cuda,
             training_data,
             threshould_iou, ratio_margin_iou, threshold_prob):
    npix = net_d.npix

    net_d.eval()
    net_c.eval()
    net_l.eval()
    sm = nn.Softmax(dim=1)

    for itr in range(10000):
        list_path_mag, nb = training_data.get_batch()
        np_tnsr, np_trgc, np_trgl, np_mask = face.get_batch(list_path_mag, 32, nb, threshould_iou, ratio_margin_iou)
        np_img = numpy.moveaxis(np_tnsr.reshape(3, nb[0] * npix, nb[1] * npix), 0, 2)
        np_img = (np_img * 255.0).astype(numpy.uint8)
        np_img = np_img.copy()
        #		assert ol.shape[0] == nblock and ol.shape[1] == nblock

        ###
        input_img = Variable(torch.from_numpy(np_tnsr), requires_grad=False)
        if is_cuda:
            input_img = input_img.cuda()
        ###
        descriptor = net_d(input_img)
        # print(descriptor.shape)
        output_c = net_c(descriptor)
        output_c = sm(output_c)
        output_l = net_l(descriptor)
        if is_cuda:
            np_outc = output_c.cpu().data.numpy()
            np_outl = output_l.cpu().data.numpy()
        else:
            np_outc = output_c.data.numpy()
            np_outl = output_l.data.numpy()
        ####
        for ih in range(nb[0]):
            for iw in range(nb[1]):
                if np_outc[0, 1, ih, iw] > threshold_prob:
                    rect0 = [npix * iw, npix * ih, npix, npix]
                    dx = np_outl[0, 0, ih, iw]
                    dy = np_outl[0, 1, ih, iw]
                    ds = np_outl[0, 2, ih, iw]
                    r2 = math.pow(2, ds * 2) * npix
                    r0 = (dx + iw + 0.5) * npix - r2 * 0.5
                    r1 = (dy + ih + 0.5) * npix - r2 * 0.5
                    rect1 = [r0, r1, r2, r2]
                    face.highlight_rect(np_img, rect0, (255, 255, 0), width=1)  # output class
                    face.highlight_rect(np_img, rect1, (0, 255, 255))  # output rect
                if np_trgc[0, ih, iw] == 1:  # target
                    rect1 = [npix * iw + 2, npix * ih + 2, npix, npix]
                    face.highlight_rect(np_img, rect1, (255, 0, 255), width=1)
        ####
        path_csv = list_path_mag[0][0]
        print(training_data.iepoch,training_data.ibatch, len(training_data.list_batch),path_csv)
        list_rect = face.get_list_rect_from_dict_info(face.get_csv_data(path_csv))
        for rect in list_rect:
            mag = list_path_mag[0][1]
            rect = [x / mag for x in rect]
            rect[0] -= list_path_mag[0][3]
            rect[1] -= list_path_mag[0][2]
            face.highlight_rect(np_img, rect, (0, 0, 255))
        ####
        cv2.imshow('img', np_img)
        ikey = int(cv2.waitKey(1000))
        print(ikey)
        if ikey == 113:
            exit()


def main():
    print(os.uname())
    list_path_dir = []
    if os.uname()[1] == "nubuntu":
        list_path_dir.append('/media/nobuyuki/D/projects/face/umet55')


    print("number of cvs file:",len(list_path_dir))
    if len(list_path_dir) == 0:
        print("Error! -> there is no input CSV file. Exiting")
        exit()

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        print("using GPU")
    else:
        print("using CPU")

    net_d, net_c, net_l = face.load_detection_network(is_cuda, '.')

    threshold_iou = 0.18  # low->hit large area (more noise?)
    ratio_margin_iou = 0.5  # low->large margin (more noise?)
    threshold_prob = 0.8
    if len(os.sys.argv) == 2 and os.sys.argv[1] == "1":
        training_data = TrainingData(list_path_dir, 1)
        evaluate(net_d, net_c, net_l, is_cuda,
                 training_data,
                 threshold_iou, ratio_margin_iou, threshold_prob)
    if len(os.sys.argv) == 2 and os.sys.argv[1] == "2":
        training_data = TrainingData(list_path_dir, 1)
        view_train(training_data, threshold_iou, ratio_margin_iou)
    else:
        training_data = TrainingData(list_path_dir, 30)
        for itr in range(400):
            print(itr)
            train(net_d, net_c, net_l, is_cuda, 1000, True, True, True,
                  training_data,
                  threshold_iou, ratio_margin_iou)


if __name__ == "__main__":
    main()
