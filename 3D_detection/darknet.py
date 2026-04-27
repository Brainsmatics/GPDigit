'''
Construction of a 3D plaque detection model.
'''

from __future__ import division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from util import predict_transform
from HBNet import hb_net
import tifffile
# dropblock
drop_if = 1

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))  # Resize to the input dimension
    img_ = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis, :, :, :] / 255.0  # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = Variable(img_)  # Convert to Variable
    return img_


def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class DropBlock3D(nn.Module):
    """
    Randomly zeroes 3D spatial blocks of the input tensor.
    An extension to the concept described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, D, H, W)`
        - Output: `(N, C, D, H, W)`
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock3D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, depth, height, width)

        assert x.dim() == 5, \
            "Expected input with 5 dimensions (bsize, channels, depth, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)
            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            # place mask on input device
            mask = mask.to(x.device)
            # compute block mask
            block_mask = self._compute_block_mask(mask)
            # apply block mask
            out = x * block_mask[:, None, :, :, :]
            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool3d(input=mask[:, None, :, :, :],
                                  kernel_size=(self.block_size, self.block_size, self.block_size),
                                  stride=(1, 1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 3)



class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x



def create_modules(blocks):
    net_info = blocks[0]  # Captures the information about the input and pre-processing
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # check the type of block
        # create a new module for the block
        # append to module_list

        # If it's a convolutional layer
        if (x["type"] == "convolutional"):
            # Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv3d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm3d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

            # 测试使用mish激活的情况
            if activation == "mish":
                activn = Mish()
                module.add_module("mish_{0}".format(index), activn)

            # If it's an upsampling layer
            # We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        # If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            # Start  of a route
            start = int(x["layers"][0])
            # end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x["type"] == "dropblock":
            drop_prob = float(x["drop_prob"])
            block_size = int(x["block_size"])
            dropblock = DropBlock3D(drop_prob, block_size)
            module.add_module("dropblock_{}".format(index), dropblock)
            # batch_normalize = int(x["batch_normalize"])
            # if batch_normalize:
            #     bn = nn.BatchNorm3d(filters)
            # module.add_module("batch_norm_{0}".format(index), bn)

        # Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1], anchors[i + 2]) for i in range(0, len(anchors), 3)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


# feature map
def to_image(input_tensor, ii):
    #print("input_tensor_size===>", input_tensor.shape)
    num = len(input_tensor[0, :])
    print(num)
    for i in range(num):
        img_out = input_tensor[0, i]
        #print(img_out.shape)
        img_out = img_out.squeeze()
        #print(img_out.shape)
        #img_out = img_out.unsqueeze(dim = 0)
        #print(img_out.shape)
        #with torch.no_grad():
        img_out_clone = img_out.clone()
        img_out_clone = img_out_clone.detach().cpu().numpy()
        img_out_clone = img_out_clone * (255)
        #print(img_out_clone)
        save_path = "./fm-show/fm/"
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        tifffile.imsave(save_path + str(ii) + "_layer_" + str(i) + ".tif", img_out_clone)



class Darknet_old(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet_old, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        # inp_dim = min(int(self.net_info["width"]), int(self.net_info["height"]))
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated() / 1024 ** 3
        print(f"Before forward: {start_mem:.2f}GB")

        modules = self.blocks[1:]
        outputs = {}  # We cache the outputs for the route layer

        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample" or module_type == "dropblock":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                # inp_dim = int(self.net_info["height"])
                inp_dim = min(int(self.net_info["width"]), int(self.net_info["height"]))

                # Get the number of classes
                num_classes = int(module["classes"])

                if i == 87:
                    scale_pos = 'scale1_pos1'
                elif i == 99:
                    scale_pos = 'scale2_pos1'
                elif i == 111:
                    scale_pos = 'scale3_pos1'
                else:
                    scale_pos = f'scale_unknown_{i}'


                # FH detect (use HBNet)
                # hb = hbn.forward(outputs[i-6], outputs[i-4], outputs[i-2])
                hb = hb_net(outputs[i - 6], outputs[i - 4], outputs[i - 2], scale_pos)
                # print(f"hb stats: min={hb.min().item()}, max={hb.max().item()}, mean={hb.mean().item()}")
                if torch.isnan(hb).any() or torch.isinf(hb).any():
                    # print(f"NaN/Inf in hb at {scale_pos}")
                    import pdb
                    pdb.set_trace()
                x_hb = predict_transform(hb, inp_dim, anchors, num_classes, CUDA)

                # Transform
                # x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                # if not write and i > 100:  # if no collector has been intialised.

                scale_output = torch.cat((x, x_hb), 1)  # 形状: [batch, 198, grid, grid, grid]

                if not write:
                    detections = []
                    detections.append(scale_output)
                    # detections.append(x)
                    write = 1
                else:
                    detections.append(scale_output)
                    # detections.append(x)
                #     pass

            outputs[i] = x
            # to_image(x, i)

        # 是否dropblock
        # print(f"detections shape: {detections[0].shape if isinstance(detections, list) else detections.shape}")
        end_mem = torch.cuda.memory_allocated() / 1024 ** 3
        print(f"After forward: {end_mem:.2f}GB, increase: {end_mem - start_mem:.2f}GB")

        return detections

    def load_weights(self, weightfile):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # If module_type is convolutional load weights
            # Otherwise ignore.

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

        self.anchors_per_scale = []
        self.scale_indices = []
        self.scale_names = ['scale1', 'scale2', 'scale3']

        yolo_count = 0
        for i, module in enumerate(self.module_list):
            if len(module) > 0 and hasattr(module[0], 'anchors'):
                self.anchors_per_scale.append(module[0].anchors)
                self.scale_indices.append(i)
                yolo_count += 1


        # for idx, (i, anchors) in enumerate(zip(self.scale_indices, self.anchors_per_scale)):
        #     print(f"scale{idx}: index{i}, {len(anchors)}个anchors")

    def forward(self, x, CUDA):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated() / 1024 ** 3
            # print(f"Darknet forward: {start_mem:.2f}GB")

        modules = self.blocks[1:]
        outputs = {}
        detections = []

        for i, module in enumerate(modules):
            module_type = module["type"]

            if module_type in ["convolutional", "upsample", "dropblock"]:
                x = self.module_list[i](x)

            elif module_type == "route":

                raw_layers = module["layers"]
                if isinstance(raw_layers, str):
                    layers = [int(x.strip()) for x in raw_layers.split(',')]
                else:
                    layers = [int(x) for x in raw_layers]

                if len(layers) == 1:
                    start = layers[0]
                    if start > 0:
                        start = start - i
                    x = outputs[i + start]
                else:
                    start = layers[0]
                    end = layers[1]
                    if start > 0:
                        start = start - i
                    if end > 0:
                        end = end - i

                    map1 = outputs[i + start]
                    map2 = outputs[i + end]
                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])

                scale_idx = len(detections)
                scale_name = self.scale_names[scale_idx]

                # HBNet
                hb_inputs = []
                input_indices = []
                for offset in [-6, -4, -2]:
                    target_idx = i + offset
                    if target_idx >= 0 and target_idx in outputs:
                        hb_inputs.append(outputs[target_idx])
                        input_indices.append(target_idx)



                if len(hb_inputs) == 3:
                    # use HBNet
                    hb = hb_net(hb_inputs[0], hb_inputs[1], hb_inputs[2], scale_name)
                    if torch.isnan(hb).any() or torch.isinf(hb).any():
                        hb = torch.zeros_like(x)
                    x_hb = predict_transform(hb, inp_dim, anchors, num_classes, CUDA)
                else:
                    x_hb = torch.zeros_like(x)

                # transform
                x_main = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                # x_main: [batch, 99, grid, grid, grid]
                # x_hb: [batch, 99, grid, grid, grid]
                # combined: [batch, 198, grid, grid, grid]
                combined = torch.cat((x_main, x_hb), dim=1)

                detections.append(combined)

            outputs[i] = x

        if torch.cuda.is_available():
            end_mem = torch.cuda.memory_allocated() / 1024 ** 3
            peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            # print(f"Darknet forward: {end_mem:.2f}GB,  {end_mem - start_mem:.2f}GB,  {peak_mem:.2f}GB")

        return detections