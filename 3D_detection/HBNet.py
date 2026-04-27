import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device('cuda:0')
#
# def hb_12(conv1, conv2):
#     X = conv1 * conv2
#     X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
#     X = torch.nn.functional.normalize(X)
#     return X
#
# def forward(f_conv1, f_conv2, f_conv3):
#     X_conv_1 = f_conv1
#     X_conv_2 = f_conv2
#     X_conv_3 = f_conv3
#
#     X_branch_1 = hb_12(X_conv_1, X_conv_2)
#     X_branch_2 = hb_12(X_conv_1, X_conv_3)
#     X_branch_3 = hb_12(X_conv_2, X_conv_3)
#
#     X_branch = torch.cat([X_branch_1, X_branch_2, X_branch_3], dim=1)
#     prev_filters = X_branch.size(1)
#     conv_end = nn.Conv3d(prev_filters, 99, kernel_size=1, stride=1, padding=0, bias=True)
#     conv_end = conv_end.to(device)
#     hb_out = conv_end(X_branch)
#     return hb_out


# class HBNet(nn.Module):
#     def __init__(self, out_channels=99):
#         super(HBNet, self).__init__()
#         self.out_channels = out_channels
#         # 为每个尺度和每个位置创建独立的conv层
#         self.conv_layers = nn.ModuleDict()
#
#     def hb_12(self, conv1, conv2):
#         X = conv1 * conv2
#         X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
#         X = F.normalize(X)
#         X = torch.clamp(X, min=-1, max=1)  # 限制输出范围
#         return X
#
#     def forward(self, f_conv1, f_conv2, f_conv3, scale_pos):
#         """
#         scale_pos: 例如 'scale1_pos1', 'scale2_pos2' 等
#         """
#         X_branch_1 = self.hb_12(f_conv1, f_conv2)
#         X_branch_2 = self.hb_12(f_conv1, f_conv3)
#         X_branch_3 = self.hb_12(f_conv2, f_conv3)
#
#         # 👇 检查每个分支
#         for name, tensor in [('X_branch_1', X_branch_1), ('X_branch_2', X_branch_2), ('X_branch_3', X_branch_3)]:
#             if torch.isnan(tensor).any():
#                 print(f"NaN in {name} at {scale_pos}")
#                 import pdb;
#                 pdb.set_trace()
#             if torch.isinf(tensor).any():
#                 print(f"Inf in {name} at {scale_pos}")
#                 import pdb;
#                 pdb.set_trace()
#
#         X_branch = torch.cat([X_branch_1, X_branch_2, X_branch_3], dim=1)
#
#         if torch.isnan(X_branch).any():
#             print(f"NaN in X_branch after cat at {scale_pos}")
#             import pdb;
#             pdb.set_trace()
#         if torch.isinf(X_branch).any():
#             print(f"Inf in X_branch after cat at {scale_pos}")
#             import pdb;
#             pdb.set_trace()
#
#         # 为每个位置创建独立的conv层
#         if scale_pos not in self.conv_layers:
#             in_channels = X_branch.size(1)
#             self.conv_layers[scale_pos] = nn.Conv3d(
#                 in_channels, self.out_channels,
#                 kernel_size=1, stride=1, padding=0, bias=True
#             ).to(X_branch.device)
#             print(f"创建HBNet层: {scale_pos}, 输入通道: {in_channels}")
#
#         hb_out = self.conv_layers[scale_pos](X_branch)
#         hb_out = torch.clamp(hb_out, min=-1, max=1)  # 再加一道保险
#
#         # 👇 检查输出
#         if torch.isnan(hb_out).any():
#             print(f"NaN in hb_out at {scale_pos}")
#             import pdb;
#             pdb.set_trace()
#         if torch.isinf(hb_out).any():
#             print(f"Inf in hb_out at {scale_pos}")
#             import pdb;
#             pdb.set_trace()
#
#         return hb_out
#
#
# hb_net = HBNet(out_channels=99)




# class HBNet(nn.Module):
#     def __init__(self, out_channels=99, eps=1e-6):
#         super(HBNet, self).__init__()
#         self.out_channels = out_channels
#         self.eps = eps
#         # 为每个尺度和每个位置创建独立的conv层
#         self.conv_layers = nn.ModuleDict()
#
#     def hb_12(self, conv1, conv2, scale_pos, branch_name):
#         """
#         改进的hb_12，添加数值稳定处理
#         """
#         # 1. 先限制输入范围，防止乘积爆炸
#         conv1 = torch.clamp(conv1, min=-10, max=10)
#         conv2 = torch.clamp(conv2, min=-10, max=10)
#
#         # 2. 逐元素乘积
#         X = conv1 * conv2
#
#         # 3. 检查乘积结果
#         if torch.isnan(X).any() or torch.isinf(X).any():
#             # print(f"⚠️ NaN/Inf after multiplication in {branch_name} at {scale_pos}")
#             # print(f"conv1 range: [{conv1.min():.3f}, {conv1.max():.3f}]")
#             # print(f"conv2 range: [{conv2.min():.3f}, {conv2.max():.3f}]")
#             # 用安全值替换
#             X = torch.where(torch.isnan(X) | torch.isinf(X),
#                             torch.zeros_like(X), X)
#
#         # 4. 改进的开方操作
#         abs_X = torch.abs(X)
#         # 限制最大值，防止开方后爆炸
#         abs_X = torch.clamp(abs_X, max=100)
#
#         # 计算带符号的平方根
#         sqrt_term = torch.sqrt(abs_X + self.eps)
#         X = torch.sign(X) * sqrt_term
#
#         # 5. 再次检查
#         if torch.isnan(X).any() or torch.isinf(X).any():
#             # print(f"⚠️ NaN/Inf after sqrt in {branch_name} at {scale_pos}")
#             X = torch.where(torch.isnan(X) | torch.isinf(X),
#                             torch.zeros_like(X), X)
#
#         # 6. 归一化前限制范围
#         X = torch.clamp(X, min=-10, max=10)
#
#         # 7. 归一化
#         X = F.normalize(X, p=2, dim=1, eps=self.eps)
#
#         # 8. 最终输出限制
#         X = torch.clamp(X, min=-1, max=1)
#
#         return X
#
#     def forward(self, f_conv1, f_conv2, f_conv3, scale_pos):
#         """
#         scale_pos: 例如 'scale1_pos1', 'scale2_pos2' 等
#         """
#         # 预先限制输入
#         f_conv1 = torch.clamp(f_conv1, min=-10, max=10)
#         f_conv2 = torch.clamp(f_conv2, min=-10, max=10)
#         f_conv3 = torch.clamp(f_conv3, min=-10, max=10)
#
#         # 计算三个分支，传入scale_pos和分支名
#         X_branch_1 = self.hb_12(f_conv1, f_conv2, scale_pos, 'branch1')
#         X_branch_2 = self.hb_12(f_conv1, f_conv3, scale_pos, 'branch2')
#         X_branch_3 = self.hb_12(f_conv2, f_conv3, scale_pos, 'branch3')
#
#         # 👇 检查每个分支
#         for name, tensor in [('X_branch_1', X_branch_1), ('X_branch_2', X_branch_2), ('X_branch_3', X_branch_3)]:
#             if torch.isnan(tensor).any():
#                 # print(f"❌ NaN in {name} at {scale_pos}")
#                 # 打印统计信息
#                 # print(f"   range: [{tensor.min():.3f}, {tensor.max():.3f}]")
#                 # print(f"   mean: {tensor.mean():.3f}, std: {tensor.std():.3f}")
#                 # 用零替换问题值，而不是进入pdb
#                 tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
#             if torch.isinf(tensor).any():
#                 # print(f"❌ Inf in {name} at {scale_pos}")
#                 tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
#
#         X_branch = torch.cat([X_branch_1, X_branch_2, X_branch_3], dim=1)
#
#         if torch.isnan(X_branch).any():
#             # print(f"❌ NaN in X_branch after cat at {scale_pos}")
#             X_branch = torch.where(torch.isnan(X_branch), torch.zeros_like(X_branch), X_branch)
#         if torch.isinf(X_branch).any():
#             # print(f"❌ Inf in X_branch after cat at {scale_pos}")
#             X_branch = torch.where(torch.isinf(X_branch), torch.zeros_like(X_branch), X_branch)
#
#         # 为每个位置创建独立的conv层
#         if scale_pos not in self.conv_layers:
#             in_channels = X_branch.size(1)
#             self.conv_layers[scale_pos] = nn.Conv3d(
#                 in_channels, self.out_channels,
#                 kernel_size=1, stride=1, padding=0, bias=True
#             ).to(X_branch.device)
#             # print(f"✅ 创建HBNet层: {scale_pos}, 输入通道: {in_channels}")
#
#         hb_out = self.conv_layers[scale_pos](X_branch)
#
#         # 检查卷积输出
#         if torch.isnan(hb_out).any() or torch.isinf(hb_out).any():
#             # print(f"❌ NaN/Inf in conv output at {scale_pos}")
#             # 查看卷积层权重
#             conv_weight = self.conv_layers[scale_pos].weight
#             # print(f"conv weight range: [{conv_weight.min():.3f}, {conv_weight.max():.3f}]")
#             hb_out = torch.where(torch.isnan(hb_out) | torch.isinf(hb_out),
#                                  torch.zeros_like(hb_out), hb_out)
#
#         hb_out = torch.clamp(hb_out, min=-1, max=1)  # 再加一道保险
#
#         # 👇 检查输出
#         if torch.isnan(hb_out).any():
#             # print(f"❌ NaN in hb_out at {scale_pos}")
#             hb_out = torch.where(torch.isnan(hb_out), torch.zeros_like(hb_out), hb_out)
#         if torch.isinf(hb_out).any():
#             # print(f"❌ Inf in hb_out at {scale_pos}")
#             hb_out = torch.where(torch.isinf(hb_out), torch.zeros_like(hb_out), hb_out)
#
#         # # 打印统计信息（可选，用于监控）
#         # if torch.rand(1).item() < 0.01:  # 1%的概率打印
#         #     print(f"📊 {scale_pos} stats: min={hb_out.min():.3f}, max={hb_out.max():.3f}, mean={hb_out.mean():.3f}")
#
#         return hb_out

class HBNet(nn.Module):
    def __init__(self, out_channels=99, eps=1e-6):
        super(HBNet, self).__init__()
        self.out_channels = out_channels
        self.eps = eps
        # 为每个尺度创建独立的conv层（只有3个尺度，每个尺度1个位置）
        self.conv_layers = nn.ModuleDict()

    def hb_12(self, conv1, conv2, scale_name):
        """
        改进的hb_12，添加数值稳定处理
        """
        # 1. 先限制输入范围
        conv1 = torch.clamp(conv1, min=-10, max=10)
        conv2 = torch.clamp(conv2, min=-10, max=10)

        # 2. 逐元素乘积
        X = conv1 * conv2

        # 3. 检查乘积结果
        if torch.isnan(X).any() or torch.isinf(X).any():
            X = torch.where(torch.isnan(X) | torch.isinf(X),
                            torch.zeros_like(X), X)

        # 4. 开方操作
        abs_X = torch.abs(X)
        abs_X = torch.clamp(abs_X, max=100)
        sqrt_term = torch.sqrt(abs_X + self.eps)
        X = torch.sign(X) * sqrt_term

        # 5. 再次检查
        if torch.isnan(X).any() or torch.isinf(X).any():
            X = torch.where(torch.isnan(X) | torch.isinf(X),
                            torch.zeros_like(X), X)

        # 6. 归一化前限制范围
        X = torch.clamp(X, min=-10, max=10)

        # 7. 归一化
        X = F.normalize(X, p=2, dim=1, eps=self.eps)

        # 8. 最终输出限制
        X = torch.clamp(X, min=-1, max=1)

        return X

    def forward(self, f_conv1, f_conv2, f_conv3, scale_name):
        """
        scale_name: 'scale1', 'scale2', 'scale3'
        """
        # 预先限制输入
        f_conv1 = torch.clamp(f_conv1, min=-10, max=10)
        f_conv2 = torch.clamp(f_conv2, min=-10, max=10)
        f_conv3 = torch.clamp(f_conv3, min=-10, max=10)

        # 计算三个分支
        X_branch_1 = self.hb_12(f_conv1, f_conv2, scale_name)
        X_branch_2 = self.hb_12(f_conv1, f_conv3, scale_name)
        X_branch_3 = self.hb_12(f_conv2, f_conv3, scale_name)

        # 检查每个分支
        for tensor in [X_branch_1, X_branch_2, X_branch_3]:
            if torch.isnan(tensor).any():
                tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
            if torch.isinf(tensor).any():
                tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)

        # 拼接三个分支
        X_branch = torch.cat([X_branch_1, X_branch_2, X_branch_3], dim=1)

        # 检查拼接结果
        if torch.isnan(X_branch).any():
            X_branch = torch.where(torch.isnan(X_branch), torch.zeros_like(X_branch), X_branch)
        if torch.isinf(X_branch).any():
            X_branch = torch.where(torch.isinf(X_branch), torch.zeros_like(X_branch), X_branch)

        # 为每个尺度创建独立的conv层
        if scale_name not in self.conv_layers:
            in_channels = X_branch.size(1)
            self.conv_layers[scale_name] = nn.Conv3d(
                in_channels, self.out_channels,
                kernel_size=1, stride=1, padding=0, bias=True
            ).to(X_branch.device)

            # 初始化权重，避免梯度爆炸
            nn.init.xavier_uniform_(self.conv_layers[scale_name].weight, gain=0.1)
            nn.init.zeros_(self.conv_layers[scale_name].bias)

        hb_out = self.conv_layers[scale_name](X_branch)

        # 检查卷积输出
        if torch.isnan(hb_out).any() or torch.isinf(hb_out).any():
            hb_out = torch.where(torch.isnan(hb_out) | torch.isinf(hb_out),
                                 torch.zeros_like(hb_out), hb_out)

        # 使用tanh限制输出范围，与sigmoid匹配
        hb_out = torch.tanh(hb_out)

        return hb_out

hb_net = HBNet(out_channels=99)
