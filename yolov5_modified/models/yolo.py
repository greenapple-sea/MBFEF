# # YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# """
# YOLO-specific modules
#
# Usage:
#     $ python path/to/models/yolo.py --cfg yolov5s.yaml
# """
#
# import argparse
# import sys
# from copy import deepcopy
# from pathlib import Path
#
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[1]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative
#
# from models.common import *
# from models.experimental import *
# from utils.autoanchor import check_anchor_order
# from utils.general import check_yaml, make_divisible, print_args, set_logging
# from utils.plots import feature_visualization
# from utils.torch_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
#     select_device, time_sync
#
# try:
#     import thop  # for FLOPs computation
# except ImportError:
#     thop = None
#
# LOGGER = logging.getLogger(__name__)
#
#
# class Detect(nn.Module):
#     stride = None  # strides computed during build
#     onnx_dynamic = False  # ONNX export parameter
#
#     def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.no = nc + 5  # number of outputs per anchor
#         self.nl = len(anchors)  # number of detection layers
#         self.na = len(anchors[0]) // 2  # number of anchors
#         self.grid = [torch.zeros(1)] * self.nl  # init grid
#         a = torch.tensor(anchors).float().view(self.nl, -1, 2)
#         self.register_buffer('anchors', a)  # shape(nl,na,2)
#         self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
#         self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
#         self.inplace = inplace  # use in-place ops (e.g. slice assignment)
#
#     def forward(self, x):
#         z = []  # inference output
#         for i in range(self.nl):
#             x[i] = self.m[i](x[i])  # conv
#             bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
#             x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#
#             if not self.training:  # inference
#                 if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
#                     self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
#
#                 y = x[i].sigmoid()
#                 if self.inplace:
#                     y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
#                     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
#                 else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
#                     xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
#                     wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
#                     y = torch.cat((xy, wh, y[..., 4:]), -1)
#                 z.append(y.view(bs, -1, self.no))
#
#         return x if self.training else (torch.cat(z, 1), x)
#
#     @staticmethod
#     def _make_grid(nx=20, ny=20):
#         yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
#         return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
#
#
# class Model(nn.Module):
#     def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
#         super().__init__()
#         if isinstance(cfg, dict):
#             self.yaml = cfg  # model dict
#         else:  # is *.yaml
#             import yaml  # for torch hub
#             self.yaml_file = Path(cfg).name
#             with open(cfg) as f:
#                 self.yaml = yaml.safe_load(f)  # model dict
#
#         # Define model
#         ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
#         if nc and nc != self.yaml['nc']:
#             LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
#             self.yaml['nc'] = nc  # override yaml value
#         if anchors:
#             LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
#             self.yaml['anchors'] = round(anchors)  # override yaml value
#         self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
#         self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
#         self.inplace = self.yaml.get('inplace', True)
#
#         # Build strides, anchors
#         m = self.model[-1]  # Detect()
#         if isinstance(m, Detect):
#             s = 256  # 2x min stride
#             m.inplace = self.inplace
#             m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
#             m.anchors /= m.stride.view(-1, 1, 1)
#             check_anchor_order(m)
#             self.stride = m.stride
#             self._initialize_biases()  # only run once
#
#         # Init weights, biases
#         initialize_weights(self)
#         self.info()
#         LOGGER.info('')
#
#     def forward(self, x, augment=False, profile=False, visualize=False):
#         if augment:
#             return self._forward_augment(x)  # augmented inference, None
#         return self._forward_once(x, profile, visualize)  # single-scale inference, train
#
#     def _forward_augment(self, x):
#         img_size = x.shape[-2:]  # height, width
#         s = [1, 0.83, 0.67]  # scales
#         f = [None, 3, None]  # flips (2-ud, 3-lr)
#         y = []  # outputs
#         for si, fi in zip(s, f):
#             xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
#             yi = self._forward_once(xi)[0]  # forward
#             # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
#             yi = self._descale_pred(yi, fi, si, img_size)
#             y.append(yi)
#         return torch.cat(y, 1), None  # augmented inference, train
#
#     def _forward_once(self, x, profile=False, visualize=False):
#         y, dt = [], []  # outputs
#         for m in self.model:
#             if m.f != -1:  # if not from previous layer
#                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
#             if profile:
#                 self._profile_one_layer(m, x, dt)
#             x = m(x)  # run
#             y.append(x if m.i in self.save else None)  # save output
#             if visualize:
#                 feature_visualization(x, m.type, m.i, save_dir=visualize)
#         return x
#
#
#     def _descale_pred(self, p, flips, scale, img_size):
#         # de-scale predictions following augmented inference (inverse operation)
#         if self.inplace:
#             p[..., :4] /= scale  # de-scale
#             if flips == 2:
#                 p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
#             elif flips == 3:
#                 p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
#         else:
#             x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
#             if flips == 2:
#                 y = img_size[0] - y  # de-flip ud
#             elif flips == 3:
#                 x = img_size[1] - x  # de-flip lr
#             p = torch.cat((x, y, wh, p[..., 4:]), -1)
#         return p
#
#     def _profile_one_layer(self, m, x, dt):
#         c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
#         o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
#         t = time_sync()
#         for _ in range(10):
#             m(x.copy() if c else x)
#         dt.append((time_sync() - t) * 100)
#         if m == self.model[0]:
#             LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
#         LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
#         if c:
#             LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")
#
#     def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
#         # https://arxiv.org/abs/1708.02002 section 3.3
#         # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
#         m = self.model[-1]  # Detect() module
#         for mi, s in zip(m.m, m.stride):  # from
#             b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
#             b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
#             b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
#             mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
#
#     def _print_biases(self):
#         m = self.model[-1]  # Detect() module
#         for mi in m.m:  # from
#             b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
#             LOGGER.info(
#                 ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))
#
#     # def _print_weights(self):
#     #     for m in self.model.modules():
#     #         if type(m) is Bottleneck:
#     #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights
#
#     def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
#         LOGGER.info('Fusing layers... ')
#         for m in self.model.modules():
#             if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
#                 m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
#                 delattr(m, 'bn')  # remove batchnorm
#                 m.forward = m.forward_fuse  # update forward
#         self.info()
#         return self
#
#     def autoshape(self):  # add AutoShape module
#         LOGGER.info('Adding AutoShape... ')
#         m = AutoShape(self)  # wrap model
#         copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
#         return m
#
#     def info(self, verbose=False, img_size=640):  # print model information
#         model_info(self, verbose, img_size)
#
# # models/yolo.py 头部添加以下导入
# from models.common import MultiBranchConv # 新增
#
#
# def flatten(f):
#     if isinstance(f, list) and len(f) == 1:
#         return flatten(f[0])
#     elif isinstance(f, list):
#         return [flatten(x) for x in f]
#     else:
#         return f
# ###原
#
# def parse_model(d, ch):  # model_dict, input_channels(3)
#     LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
#     anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
#     na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
#     no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
#
#     layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
#     for i, (f, n, m, args) in enumerate(d['backbone'] +d['neck']+ d['head']):  # from, number, module, args
#         m = eval(m) if isinstance(m, str) else m  # eval strings
#         for j, a in enumerate(args):
#             try:
#                 args[j] = eval(a) if isinstance(a, str) else a  # eval strings
#             except:
#                 pass
#         if isinstance(f, int):
#             f = [f]
#
#             # 处理多输入源的特殊情况
#             # 处理MambaHRFusion特殊逻辑
#         if m == MambaHRFusion:
#             # 输入通道来自层7、5、3
#             c1 = [ch[x] for x in f]  # [1024, 512, 256]
#             # 合并参数: [in_p5, in_p4, in_p3, out_p3, out_p4, out_p5]
#             merged_args = c1 + args  # [1024,512,256,256,512,1024]
#             # 记录输出通道到ch列表
#             ch.extend(merged_args[-3:])  # [256,512,1024]
#             # 初始化模块
#             m_ = MambaHRFusion(*merged_args)
#         n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
#         if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
#                  BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
#             c1, c2 = ch[f[-1]], args[0]
#             if c2 != no:  # if not output
#                 c2 = make_divisible(c2 * gw, 8)
#
#             args = [c1, c2, *args[1:]]
#             if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
#                 args.insert(2, n)  # number of repeats
#                 n = 1
#         elif m is nn.BatchNorm2d:
#             args = [ch[f]]
#         elif m is Concat:
#             c2 = sum([ch[x] for x in f])
#         elif m is Detect:
#             args.append([ch[x] for x in f])
#             if isinstance(args[1], int):  # number of anchors
#                 args[1] = [list(range(args[1] * 2))] * len(f)
#         elif m is Contract:
#             c2 = ch[f] * args[0] ** 2
#         elif m is Expand:
#             c2 = ch[f] // args[0] ** 2
#         elif m is MultiBranchConv:
#             c1 = ch[f]  # 输入通道数
#             c2 = args[0]  # 输出通道数
#             stride = args[1] if len(args) > 1 else 1  # 步长（默认1）
#             args = [c1, c2, stride]  # 参数格式 [in, out, stride]
#         else:
#             c2 = ch[f[-1]]
#
#         m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
#         t = str(m)[8:-2].replace('__main__.', '')  # module type
#         np = sum([x.numel() for x in m_.parameters()])  # number params
#         m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
#         LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n_, np, t, args))  # print
#         save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
#         layers.append(m_)
#         if i == 0:
#             ch = []
#         ch.append(c2)
#     return nn.Sequential(*layers), sorted(save)
#
#
#
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--profile', action='store_true', help='profile model speed')
#     opt = parser.parse_args()
#     opt.cfg = check_yaml(opt.cfg)  # check YAML
#     print_args(FILE.stem, opt)
#     set_logging()
#     device = select_device(opt.device)
#
#     # Create model
#     model = Model(opt.cfg).to(device)
#     model.train()
#
#     # Profile
#     if opt.profile:
#         img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
#         y = model(img, profile=True)
#
#     # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
#     # from torch.utils.tensorboard import SummaryWriter
#     # tb_writer = SummaryWriter('.')
#     # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
#     # tb_writer.add_graph(torch.jit.trace(model, img, strict=False


# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import check_yaml, make_divisible, print_args, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
    select_device, time_sync
import torch
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=6, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            # print(f"Feature {i} shape: {x[i].shape}")
            x[i] = self.m[i](x[i])  # conv
            # print(f"检测头 {i} 输出统计：")  # 添加调试信息
            # print("最大值:", x[i].max().item())
            # print("最小值:", x[i].min().item())
            # print("NaN数量:", torch.isnan(x[i]).sum().item())
            # print("Inf数量:", torch.isinf(x[i]).sum().item())
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # 在Detect的forward中添加断言
            assert not torch.allclose(x[i], torch.zeros_like(x[i])), "输出全零"

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))


        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            # 设置 ch 参数为 MambaNeck 的输出通道
            # m.ch = [256, 512, 1024]
            # # 重新初始化卷积层
            # m.m = nn.ModuleList(
            #     nn.Conv2d(c, m.no * m.na, 1) for c in m.ch
            # )
            # # 计算 stride
            # s = 256  # 输入尺寸
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            # m.anchors /= m.stride.view(-1, 1, 1)
            # check_anchor_order(m)
            # self.stride = m.stride
            # self._initialize_biases()
        # if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once



        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x


    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def autoshape(self):  # add AutoShape module
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

# models/yolo.py 头部添加以下导入
from models.common import MultiBranchConv # 新增
# from models.common import OutputBranch1
# from models.common import OutputBranch2
from models.common import OutputBranch

from models.common import MambaNeck
from models.common import DCSAF
from models.common import DeformFusion

def flatten(f):
    if isinstance(f, list) and len(f) == 1:
        return flatten(f[0])
    elif isinstance(f, list):
        return [flatten(x) for x in f]
    else:
        return f

def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' %
                ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except Exception:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain

        custom = False

        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            c1, c2 = ch[f], args[0]
            # if m == OutputBranch1:
            #     print(f"OutputBranch {i}: in_channels={c1}, out_channels={c2}")
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]  # 设置输入和输出通道
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        # elif m is Detect:
        #     f = [f] if isinstance(f, int) else f
        #     args.append([ch[x] for x in f])  # 输入通道应为单个整数
        #     if isinstance(args[1], int):
        #         args[1] = [list(range(args[1] * 2))]  # 单层锚点配置
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is MultiBranchConv:
            c1 = ch[f]  # 输入通道数
            c2 = args[0]  # 输出通道数
            stride = args[1] if len(args) > 1 else 1  # 步长（默认1）
            args = [c1, c2, stride]  # 参数格式 [in, out, stride]
        elif m == MambaNeck:
            if max(f) >= len(ch):
                raise IndexError(f"MambaNeck 输入索引 {f} 超出了 ch 范围 {len(ch)}，请检查 YAML 配置！")
            in_channels_list = [ch[x] for x in f]  # 例如 [256, 512, 1024]
            if len(args) > 0:
                out_channels = args[-1]
            else:
                raise ValueError("MambaNeck 的输出通道 (out_channels) 未提供")
            ch.append(out_channels)
            m_ = m(in_channels_list, out_channels)
            m_.i = f
            c2 = out_channels
            custom = True


        elif m in [DCSAF]:
            in_ch1 = ch[f[0]]  # 第一个输入层的通道数
            in_ch2 = ch[f[1]]  # 第二个输入层的通道数
            args = [in_ch1, in_ch2]  # 传递两个通道数作为参数
            m_ = m(*args)  # 构造模块
            print(f"DCSAF layer {i}: from={f}, in_ch1={in_ch1}, in_ch2={in_ch2}")  # 调试输出

        elif m in [DeformFusion]:
            in_ch1 = ch[f[0]]  # 第一个输入层的通道数
            in_ch2 = ch[f[1]]  # 第二个输入层的通道数
            args = [in_ch1, in_ch2]  # 传递两个通道数作为参数
            m_ = m(*args)  # 构造模块
            print(f"DCSAF layer {i}: from={f}, in_ch1={in_ch1}, in_ch2={in_ch2}")  # 调试输出


        elif m == OutputBranch:
            c1 = ch[f]  # 输入通道
            c2 = args[0]  # 输出通道
            args = [c1, c2, *args[1:]]  # 参数格式 [in_channels, out_channels, ...]
            m_ = m(*args)
            c2 = args[1]  # 确保 c2 是最终输出通道
            custom = False  # 或者根据需要设置 custom 标志
        # elif m == OutputBranch2:
        #     c1 = ch[f]  # 输入通道
        #     c2 = args[0]  # 输出通道
        #     args = [c1, c2, *args[1:]]  # 参数格式 [in_channels, out_channels, ...]
        #     m_ = m(*args)
        #     c2 = args[1]  # 确保 c2 是最终输出通道
        #     custom = False  # 或者根据需要设置 custom 标志
        else:
            c2 = ch[f]

        # 如果没有经过特殊处理，则走通用构造流程
        if not custom:
            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
            m_.i = f  # 记录模块输入索引

        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np_param = sum([x.numel() for x in m_.parameters()])  # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np_param  # attach index, 'from' index, type, number of parameters
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' %
                    (i, f, n_, np_param, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

# def parse_model(d, ch):  # model_dict, input_channels(3)
#     LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' %
#                 ('', 'from', 'n', 'params', 'module', 'arguments'))
#     anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
#     na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
#     no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
#
#     layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
#     for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
#         m = eval(m) if isinstance(m, str) else m  # eval strings
#         for j, a in enumerate(args):
#             try:
#                 args[j] = eval(a) if isinstance(a, str) else a  # eval strings
#             except Exception:
#                 pass
#
#         n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
#
#         # 标志变量：是否已进行特殊处理（例如 MambaNeck），以避免走通用构造流程
#         custom = False
#
#         if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
#                  BottleneckCSP, C3, C3TR, C3SPP, C3Ghost,OutputBranch]:
#             c1, c2 = ch[f], args[0]
#             if c2 != no:  # if not output
#                 c2 = make_divisible(c2 * gw, 8)
#             args = [c1, c2, *args[1:]]
#             if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
#                 args.insert(2, n)  # number of repeats
#                 n = 1
#         elif m is nn.BatchNorm2d:
#             args = [ch[f]]
#         elif m is Concat:
#             c2 = sum([ch[x] for x in f])
#         elif m is Detect:
#             # 处理单层输入的 Detect
#             f = [f] if isinstance(f, int) else f
#             args.append([ch[x] for x in f])  # 输入通道应为单个整数
#             if isinstance(args[1], int):
#                 args[1] = [list(range(args[1] * 2))]  # 单层锚点配置
#         elif m is Contract:
#             c2 = ch[f] * args[0] ** 2
#         elif m is Expand:
#             c2 = ch[f] // args[0] ** 2
#         elif m is MultiBranchConv:
#             c1 = ch[f]  # 输入通道数
#             c2 = args[0]  # 输出通道数
#             stride = args[1] if len(args) > 1 else 1  # 步长（默认1）
#             args = [c1, c2, stride]  # 参数格式 [in, out, stride]
#         elif m == MambaNeck:
#             print(f"[Debug] f = {f}, ch = {ch}")
#             if max(f) >= len(ch):
#                 raise IndexError(f"MambaNeck 输入索引 {f} 超出了 ch 范围 {len(ch)}，请检查 YAML 配置！")
#             in_channels_list = [ch[x] for x in f]  # 例如 [256, 512, 1024]
#             if len(args) > 0:
#                 out_channels = args[-1]
#             else:
#                 raise ValueError("MambaNeck 的输出通道 (out_channels) 未提供")
#             print(f"[Debug] MambaNeck 输入通道: {in_channels_list}, 输出通道: {out_channels}")
#             ch.append(out_channels)
#             m_ = m(in_channels_list, out_channels)
#             m_.i = f
#             c2 = out_channels
#             custom = True
#
#
#         else:
#             c2 = ch[f]
#
#         # 如果没有经过特殊处理，则走通用构造流程
#         if not custom:
#             m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
#             m_.i = f  # 记录模块输入索引
#
#         t = str(m)[8:-2].replace('__main__.', '')  # module type
#         np_param = sum([x.numel() for x in m_.parameters()])  # number of parameters
#         m_.i, m_.f, m_.type, m_.np = i, f, t, np_param  # attach index, 'from' index, type, number of parameters
#         LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' %
#                     (i, f, n_, np_param, t, args))  # print
#         save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
#         layers.append(m_)
#         if i == 0:
#             ch = []
#         ch.append(c2)
#     return nn.Sequential(*layers), sorted(save)


###原
# def parse_model(d, ch):  # model_dict, input_channels(3)
#     LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
#     anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
#     na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
#     no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
#
#     layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
#     for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
#         m = eval(m) if isinstance(m, str) else m  # eval strings
#         for j, a in enumerate(args):
#             try:
#                 args[j] = eval(a) if isinstance(a, str) else a  # eval strings
#             except:
#                 pass
#
#         n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
#         if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
#                  BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
#             c1, c2 = ch[f], args[0]
#             if c2 != no:  # if not output
#                 c2 = make_divisible(c2 * gw, 8)
#
#             args = [c1, c2, *args[1:]]
#             if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
#                 args.insert(2, n)  # number of repeats
#                 n = 1
#         elif m is nn.BatchNorm2d:
#             args = [ch[f]]
#         elif m is Concat:
#             c2 = sum([ch[x] for x in f])
#         elif m is Detect:
#             # 处理单层输入的Detect
#             f = [f] if isinstance(f, int) else f
#             args.append([ch[x] for x in f])  # 输入通道应为单个整数
#             if isinstance(args[1], int):
#                 args[1] = [list(range(args[1] * 2))]  # 单层锚点配置
#             # args.append([ch[x] for x in f])
#             # if isinstance(args[1], int):  # number of anchors
#             #     args[1] = [list(range(args[1] * 2))] * len(f)
#         elif m is Contract:
#             c2 = ch[f] * args[0] ** 2
#         elif m is Expand:
#             c2 = ch[f] // args[0] ** 2
#         elif m is MultiBranchConv:
#             c1 = ch[f]  # 输入通道数
#             c2 = args[0]  # 输出通道数
#             stride = args[1] if len(args) > 1 else 1  # 步长（默认1）
#             args = [c1, c2, stride]  # 参数格式 [in, out, stride]
#
#         elif m == MambaNeck:
#             # print(f"[Debug] f = {f}, ch = {ch}")
#             # c1 = [ch[x] for x in f]  # 例如 [1024, 512, 256]
#             # merged_args = c1 + args
#             # assert len(merged_args) == 4, "参数错误"
#             # ch.append(merged_args[-1])
#             # m_ = m(*merged_args)
#             # m_.i = f  # 记录输入来源索引（
#             print(f"[Debug] f = {f}, ch = {ch}")  # 打印调试信息
#
#             if max(f) >= len(ch):  # 避免索引越界
#                 raise IndexError(f"MambaNeck 输入索引 {f} 超出了 ch 范围 {len(ch)}，请检查 YAML 配置！")
#
#             # 获取输入通道列表
#             in_channels_list = [ch[x] for x in f]  # 获取输入通道，例如 [256, 512, 1024]
#
#             # 检查并传递输出通道（即args最后一个参数）
#             if len(args) > 0:
#                 out_channels = args[-1]  # 获取最后一个参数作为输出通道
#             else:
#                 raise ValueError("MambaNeck 的输出通道 (out_channels) 未提供")
#
#             print(f"[Debug] MambaNeck 输入通道: {in_channels_list}, 输出通道: {out_channels}")  # 打印调试信息
#
#             # 确保 merged_args 只包含两个正确的参数
#             merged_args = [in_channels_list, out_channels]
#
#             # 打印 merged_args 内容，确认是否传递正确
#             print(f"[Debug] merged_args: {merged_args}")
#
#             ch.append(out_channels)  # 记录输出通道，确保后续计算不会丢失
#
#             # 创建 MambaNeck 模块
#             m_ = m(in_channels_list, out_channels)  # 直接传递两个参数
#
#             m_.i = f  # 记录输入索引
#         else:
#             c2 = ch[f]
#
#         m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
#         t = str(m)[8:-2].replace('__main__.', '')  # module type
#         np = sum([x.numel() for x in m_.parameters()])  # number params
#         m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
#         LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n_, np, t, args))  # print
#         save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
#         layers.append(m_)
#         if i == 0:
#             ch = []
#         ch.append(c2)
#     return nn.Sequential(*layers), sorted(save)

# def parse_model(d, ch):  # model_dict, input_channels(3)
#     LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
#     anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
#     na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
#     no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
#
#     layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
#     for i, (f, n, m, args) in enumerate(d['backbone'] +d['neck']+ d['head']):  # from, number, module, args
#         m = eval(m) if isinstance(m, str) else m  # eval strings
#         for j, a in enumerate(args):
#             try:
#                 args[j] = eval(a) if isinstance(a, str) else a  # eval strings
#             except:
#                 pass
#         if isinstance(f, int):
#             f = [f]
#
#             # 处理多输入源的特殊情况
#             # 处理MambaHRFusion特殊逻辑
#         if m == MambaHRFusion:
#             c1 = [ch[x] for x in f]  # 输入通道列表 [P5, P4, P3]
#             args = c1 + args  # 合并输入通道和配置参数
#             c2 = args[-3:]  # 输出通道 [out_p3, out_p4, out_p5]
#         n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
#         if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
#                  BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
#             c1, c2 = ch[f], args[0]
#             if c2 != no:  # if not output
#                 c2 = make_divisible(c2 * gw, 8)
#
#             args = [c1, c2, *args[1:]]
#             if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
#                 args.insert(2, n)  # number of repeats
#                 n = 1
#         elif m is nn.BatchNorm2d:
#             args = [ch[f]]
#         elif m is Concat:
#             c2 = sum([ch[x] for x in f])
#         elif m is Detect:
#             args.append([ch[x] for x in f])
#             if isinstance(args[1], int):  # number of anchors
#                 args[1] = [list(range(args[1] * 2))] * len(f)
#         elif m is Contract:
#             c2 = ch[f] * args[0] ** 2
#         elif m is Expand:
#             c2 = ch[f] // args[0] ** 2
#         elif m is MultiBranchConv:
#             c1 = ch[f]  # 输入通道数
#             c2 = args[0]  # 输出通道数
#             stride = args[1] if len(args) > 1 else 1  # 步长（默认1）
#             args = [c1, c2, stride]  # 参数格式 [in, out, stride]
#         else:
#             c2 = ch[f]
#
#         m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
#         t = str(m)[8:-2].replace('__main__.', '')  # module type
#         np = sum([x.numel() for x in m_.parameters()])  # number params
#         m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
#         LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n_, np, t, args))  # print
#         save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
#         layers.append(m_)
#         if i == 0:
#             ch = []
#         ch.append(c2)
#     return nn.Sequential(*layers), sorted(save)
#




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False