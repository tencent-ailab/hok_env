# -*- coding: utf-8 -*-
import torch
import os
from rl_framework.common.logging import logger as LOG


class LocalTorchPredictor(object):
    def __init__(self, net):
        super().__init__()
        self.device = torch.device("cpu")
        self.net = net.to(self.device)

    def load_model(self, model_path):
        model_filename = os.path.join(model_path, "model.pth")
        LOG.info("load model: {}", model_filename)
        checkpoint = torch.load(model_filename, map_location=self.device)
        self.net.load_state_dict(checkpoint["network_state_dict"])

    def inference(self, data_list):
        torch_inputs = [
            torch.from_numpy(nparr).to(torch.float32) for nparr in data_list
        ]
        format_inputs = self.net.format_data(torch_inputs, inference=True)
        self.net.eval()
        with torch.no_grad():
            rst_list = self.net(format_inputs, inference=True)
        return rst_list
