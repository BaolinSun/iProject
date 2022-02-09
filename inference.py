import torch
import torch.nn as nn
import cv2

from model.ins_his import INS_HIS

if __name__ == '__main__':
    img = cv2.imread('tmp.jpg')
    img = cv2.resize(img, (768, 448))
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img, dtype=torch.float32)
    img = img.unsqueeze(0)
    img = img.cuda()

    model = INS_HIS()
    model = model.cuda()

    x = model.forward(img)

    print('success...')