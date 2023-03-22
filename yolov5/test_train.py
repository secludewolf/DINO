from models_yolo.common import DetectMultiBackend
from models_yolo.yolo import Model
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression
import torch

ckpt = torch.load("C:\BaiduSyncdisk\WorkSpace\Python\yolov5\ip102_best.pt")
print(ckpt.keys())
# model = Model(cfg=ckpt['model'].yaml, ch=3, nc=102, anchors=None)
# state_dict = ckpt['model'].float().state_dict()
# model.load_state_dict(state_dict, strict=False)
# model = Model(cfg="C:\BaiduSyncdisk\WorkSpace\Python\yolov5\models_yolo\yolov5s.yaml", ch=3, nc=102, anchors=None)

# model = DetectMultiBackend("C:\BaiduSyncdisk\WorkSpace\Python\yolov5\ip102_best.pt")
# print(model)
# dataset = LoadImages("C:/Users/PatrickStar/Desktop/Dataset/ip102_v1.1/Classification/images/00000.jpg", img_size=640,
#                      stride=32)
# for path, im, im0s, vid_cap, s in dataset:
#     img = torch.from_numpy(im)
#     img = img.float()
#     img /= 255.0
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)
#     pred = model(img)
#     print(pred[0])
#     print(torch.topk(pred[0], 10, 2))
    # print(pred[0].shape)
    # pred = non_max_suppression(pred, 0.1, 0.1, None, False, max_det=1000)
    # print(pred)
# output = model(torch.rand((1, 3, 512, 512)))
# print(output[0].shape)
# nms_output = non_max_suppression(output[0], 0.1, 0.1, None, False, max_det=1000)
# # print(output[0].shape)
# # print(output[1].shape)
# print(output[0].shape)
# print(nms_output[0].shape)
