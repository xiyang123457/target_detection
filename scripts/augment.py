import os

# 关掉 albumentations 的"联网检查更新"：它 import 时会请求 pypi.org，
# 网络超时/失败会让整个 import 崩掉。官方开关 NO_ALBUMENTATIONS_UPDATE=1 直接跳过。
# 【必须在 import albumentations 之前设置】
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import albumentations as A

def build_train_transform():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),      # 50% 概率水平翻转
             A.RandomScale(scale_limit=0.2, p=1.0),   # 随机缩放 0.8~1.2
        ],
         bbox_params=A.BboxParams(format="pascal_voc",label_fields=["labels"])
    )
def hflip_boxes(boxes,width):
    #水平翻转框
    # 参数 boxes: np.ndarray [N, 4] float32，每行 [x1, y1, x2, y2]
    new_x1= width -boxes[:,2] #:表示第一维取所有
    new_x2 = width -boxes[:,0]

    # 复制一份，避免污染调用方传进来的数组
    out =boxes.copy()
    out[:,0] = new_x1
    out[:,2] = new_x2

    return out

def hflip(img,boxes):
    new_img = cv2.flip(img,1)
    W =img.shape[1]  # shape 是 (H, W, 3)，宽度在下标 1，别写成 shape[0]
    return new_img,hflip_boxes(boxes,W)


def resize_boxes(boxes,old_w,old_h,new_w,new_h):
    #按比例缩放框坐标

    sx= new_w / old_w
    sy = new_h / old_h

    new_x1 = boxes[:,0] * sx
    new_y1 = boxes[:,1] * sy
    new_x2 = boxes[:,2] * sx
    new_y2 = boxes[:,3] * sy

    out = boxes.copy()
    out[:,0] = new_x1
    out[:,1] = new_y1
    out[:,2] = new_x2
    out[:,3] = new_y2

    return out

def resize_img(img,new_w,new_h):
    return cv2.resize(img,(new_w,new_h))

def resize(img,boxes,new_w,new_h):
    #同时缩放图和框
    old_h,old_w = img.shape[:2]
    new_img = resize_img(img,new_w,new_h)
    new_boxes = resize_boxes(boxes,old_w,old_h,new_w,new_h)
    return new_img,new_boxes

#测试
if __name__ == "__main__":
    boxes = np.array([[173., 100., 348., 350.]], dtype=np.float32)
    out = hflip_boxes(boxes, width=500)
    print("翻转后:", out)
    w_old = boxes[0, 2] - boxes[0, 0]
    w_new = out[0, 2] - out[0, 0]
    print(f"框宽: 原 {w_old} -> 新 {w_new}")     # 期望 175.0 -> 175.0（必须不变）

    # 顺便确认没有污染原数组
    print("原数组未被改:", boxes)                 # 期望仍是 [[173. 100. 348. 350.]]
    
