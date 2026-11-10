import torch
import torch.nn as nn


# SimAM 保持不变
class SimAM(nn.Module):
    def __init__(self, lambda_param=1e-4):
        super(SimAM, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.lambda_param)) + 0.5
        return x * torch.sigmoid(y)

class SRFE-BLOCK(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 shortcut=True):
        super(SRFE-BLOCK, self).__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)


        c_rep = ch_hidden // 2
        c_dw = ch_hidden - c_rep


        self.rep_conv = RepConv(c_rep, c_rep, 3, s=1, g=1)


        self.dw_conv = nn.Sequential(
            nn.Conv2d(c_dw, c_dw, 7, 1, 3, groups=c_dw, bias=False),
            nn.BatchNorm2d(c_dw),
            nn.SiLU()
        )

        # 4. SimAM 注意力 (融合前增强)
        self.attn = SimAM()


        self.pw_conv = Conv(ch_hidden, ch_out, 1, s=1)

        self.shortcut = shortcut

    def forward(self, x):
        # 物理分流
        x_rep, x_dw = x.chunk(2, 1)

        # 并行处理
        x_rep = self.rep_conv(x_rep)  # 捕捉细节
        x_dw = self.dw_conv(x_dw)  # 捕捉背景

        # 拼接
        y = torch.cat([x_rep, x_dw], dim=1)

        # 注意力 + 混合
        y = self.attn(y)
        y = self.pw_conv(y)

        if self.shortcut:
            return x + y
        else:
            return y


class LKC-Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 这里的 hidden_dim 可以适当减小以节省参数
        # 比如 hidden_dim = out_ch // 2
        hidden_dim = out_ch

        # 分支1: 大核 Depthwise 卷积负责空间下采样 (抗混叠)
        # 5x5 stride=2
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=5, stride=2, padding=2, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU()
        )
        # 1x1 调整通道
        self.pw_conv = Conv(in_ch, out_ch, 1, 1)

        # 分支2: MaxPool (保留显著特征)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.pool_conv = Conv(in_ch, out_ch, 1, 1)

    def forward(self, x):
        # 不进行 Split，两个分支都看完整的输入

        # Branch 1: 大核卷积提取上下文
        x1 = self.dw_conv(x)
        x1 = self.pw_conv(x1)

        # Branch 2: MaxPool 提取纹理极值
        x2 = self.pool(x)
        x2 = self.pool_conv(x2)

        # 融合 (相加比 Concat 更省参数，且符合 ResNet 思想)
        return x1 + x2

class SSDH(nn.Module):

    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        self.stems = nn.ModuleList(Conv(x, hidc, 1) for x in ch)

        self.shared_block = nn.ModuleList(
            SRFE-BLOCK(hidc, 1.0, hidc, shortcut=True) for _ in ch
        )

        self.cv2 = nn.ModuleList(nn.Conv2d(hidc, 4 * self.reg_max, 1) for _ in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(hidc, self.nc, 1) for _ in ch)

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = self.stems[i](x[i])

            # 共享特征提取
            feat = self.shared_block[i](x[i])

            # 预测
            x[i] = torch.cat((self.cv2[i](feat), self.cv3[i](feat)), 1)

        if self.training:
            return x

        # 推理逻辑 (同上)
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a.bias.data[:] = 1.0
            b.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)

    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides