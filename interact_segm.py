import torch
import argparse
import cv2
import time
import numpy as np

from glob import glob
from tqdm import tqdm
from scipy import ndimage
from config import cfg
from compose import Compose
from models.ins_his import INS_HIS
from eval_mask import run_eval_mask
from datasets.piplines import Resize, Normalize, Pad, ImageToTensor
from datasets.piplines import TestCollect, MultiScaleFlipAug, imresize

class LoadImage(object):
    def __call__(self, results):
        results['img_shape'] = results['img'].shape
        results['ori_shape'] = results['img'].shape
        return results

test_process_pipelines = [
    Resize(keep_ratio=True),
    Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    Pad(size_divisor=32),
    ImageToTensor(keys=['img']),
    TestCollect(keys=['img']),
]
Multest = MultiScaleFlipAug(transforms=test_process_pipelines, img_scale=(1333, 800), flip=False)
test_pipeline = []
test_pipeline.append(LoadImage())
test_pipeline.append(Multest)
test_pipeline = Compose(test_pipeline)

def result2image(img, result, score_thr=0.3):
    img_show = img.copy()
    h, w, _ = img.shape

    cur_result = result[0]
    seg_label = cur_result[0]
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    cate_label = cur_result[1]
    cate_label = cate_label.cpu().numpy()
    score = cur_result[2].cpu().numpy()

    vis_index = score > score_thr
    seg_label = seg_label[vis_index]
    cate_label = cate_label[vis_index]
    cate_score = score[vis_index]
    num_mask = seg_label.shape[0]

    np.random.seed(512)
    color_masks = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)
    ]

    for idx in range(num_mask):
        # idx = -(idx+1)
        if cate_label[idx] == 0:
            cur_mask = seg_label[idx]
            cur_mask = imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            if cur_mask.sum() == 0:
                continue
            color_mask = color_masks[idx]
            cur_mask_bool = cur_mask.astype(np.bool)
            img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

            cur_cate = cate_label[idx]
            cur_score = cate_score[idx]

            label_text = 'person'
            label_text += '={:.02f}'.format(cur_score)
            center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
            vis_pos = (int(center_x), int(center_y))
            cv2.putText(img_show, label_text, vis_pos, cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255))

    return img_show

def interact_segm(model_path, input_source):

    model = INS_HIS(cfg, pretrained=model_path, mode='test')
    model = model.cuda()
    img_ori = cv2.imread(input_source)
    start_time = 0
    last_time = 0


    def drawbbox(pos):
        start_time = time.time()
        ix, iy, x, y = pos[0], pos[1], pos[2], pos[3]    

        cv2.rectangle(img_ori, (ix, iy), (x, y), (0, 255, 0), 1)
        template = img_ori[iy:y, ix:x, :]


        data = dict(filename=input_source)
        data['img'] = template
        data = test_pipeline(data)
        imgs = data['img']
        img = imgs[0].cuda().unsqueeze(0)
        img_info = data['img_metas']

        
        with torch.no_grad():
            seg_result = model.forward(img=[img], img_meta=[img_info], return_loss=False)
            if seg_result[0] == None:
                print('None')
                return
            img_show = result2image(template, seg_result)
            img_ori[iy:y, ix:x, :] = img_show
            last_time = time.time()
            print('time:', last_time - start_time)
            cv2.destroyWindow("image")
            cv2.imshow('img', img_ori)
            cv2.imwrite('result.png', img_ori)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

    def mouseEvent(event, x, y, flags, param):
        global ix, iy, drawing, mode, cap, template, tempFlag
        if event == cv2.EVENT_LBUTTONDOWN:
            tempFlag = True
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            if drawing == True:
                drawing = False
                drawbbox(pos=[ix, iy, x, y])

    img = cv2.imread(input_source)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouseEvent)
    cv2.imshow("image", img)

    cv2.waitKey()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--input', type=str, default=None)
    args = parser.parse_args()
    print(args)
    interact_segm(args.model, args.input)
    

    


# =============================================================================
# img = cv2.imread("draw.jpg")

# drawing = False
# ix, iy = -1, -1
# tempFlag = False
# def draw_circle(event, x, y, flags, param):
#     global ix, iy, drawing, mode, cap, template, tempFlag
#     if event == cv2.EVENT_LBUTTONDOWN:
#         tempFlag = True
#         drawing = True
#         ix, iy = x, y
#     elif event == cv2.EVENT_LBUTTONUP:
#         if drawing == True:
#             drawing = False
#             cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 1)
#             template = img[iy:y, ix:x, :]
#             print(ix,iy,x,y)
#             cv2.destroyWindow("image")
#             cv2.imshow('img', img)
#             # cv2.imshow('template',template)

# cv2.namedWindow("image")
# cv2.setMouseCallback("image", draw_circle)
# cv2.imshow("image", img)

# cv2.waitKey()
# cv2.destroyAllWindows()