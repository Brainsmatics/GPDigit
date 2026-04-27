import torch
import torch.nn as nn
import torch.nn.functional as F


class HBNet(nn.Module):
    def __init__(self, out_channels=99, eps=1e-6):
        super(HBNet, self).__init__()
        self.out_channels = out_channels
        self.eps = eps
        self.conv_layers = nn.ModuleDict()

    def hb_12(self, conv1, conv2, scale_name):
        conv1 = torch.clamp(conv1, min=-10, max=10)
        conv2 = torch.clamp(conv2, min=-10, max=10)

        X = conv1 * conv2

        if torch.isnan(X).any() or torch.isinf(X).any():
            X = torch.where(torch.isnan(X) | torch.isinf(X),
                            torch.zeros_like(X), X)

        abs_X = torch.abs(X)
        abs_X = torch.clamp(abs_X, max=100)
        sqrt_term = torch.sqrt(abs_X + self.eps)
        X = torch.sign(X) * sqrt_term

        if torch.isnan(X).any() or torch.isinf(X).any():
            X = torch.where(torch.isnan(X) | torch.isinf(X),
                            torch.zeros_like(X), X)

        X = torch.clamp(X, min=-10, max=10)


        X = F.normalize(X, p=2, dim=1, eps=self.eps)

        X = torch.clamp(X, min=-1, max=1)

        return X

    def forward(self, f_conv1, f_conv2, f_conv3, scale_name):
        """
        scale_name: 'scale1', 'scale2', 'scale3'
        """
        f_conv1 = torch.clamp(f_conv1, min=-10, max=10)
        f_conv2 = torch.clamp(f_conv2, min=-10, max=10)
        f_conv3 = torch.clamp(f_conv3, min=-10, max=10)

        X_branch_1 = self.hb_12(f_conv1, f_conv2, scale_name)
        X_branch_2 = self.hb_12(f_conv1, f_conv3, scale_name)
        X_branch_3 = self.hb_12(f_conv2, f_conv3, scale_name)

        for tensor in [X_branch_1, X_branch_2, X_branch_3]:
            if torch.isnan(tensor).any():
                tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
            if torch.isinf(tensor).any():
                tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)

        X_branch = torch.cat([X_branch_1, X_branch_2, X_branch_3], dim=1)

        if torch.isnan(X_branch).any():
            X_branch = torch.where(torch.isnan(X_branch), torch.zeros_like(X_branch), X_branch)
        if torch.isinf(X_branch).any():
            X_branch = torch.where(torch.isinf(X_branch), torch.zeros_like(X_branch), X_branch)

        if scale_name not in self.conv_layers:
            in_channels = X_branch.size(1)
            self.conv_layers[scale_name] = nn.Conv3d(
                in_channels, self.out_channels,
                kernel_size=1, stride=1, padding=0, bias=True
            ).to(X_branch.device)

            nn.init.xavier_uniform_(self.conv_layers[scale_name].weight, gain=0.1)
            nn.init.zeros_(self.conv_layers[scale_name].bias)

        hb_out = self.conv_layers[scale_name](X_branch)

        if torch.isnan(hb_out).any() or torch.isinf(hb_out).any():
            hb_out = torch.where(torch.isnan(hb_out) | torch.isinf(hb_out),
                                 torch.zeros_like(hb_out), hb_out)

        hb_out = torch.tanh(hb_out)

        return hb_out

hb_net = HBNet(out_channels=99)
