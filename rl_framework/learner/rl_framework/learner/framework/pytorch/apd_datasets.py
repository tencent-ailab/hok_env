import torch


class Datasets(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def next(self):
        return torch.from_numpy(self.dataset.get_next_batch())

    def get_recv_speed(self):
        return self.dataset.get_recv_speed()


class DataPrefetcher:
    def __init__(self, dataset, device, use_fp16) -> None:
        self.dataset = dataset
        self.device = device
        self.use_fp16 = use_fp16
        self.next_data = None
        self.stream = torch.cuda.Stream(device=self.device)
        self.preload()

    def preload(self):
        self.next_data = self.dataset.get_next_batch()
        with torch.cuda.stream(self.stream):
            self.next_data = torch.from_numpy(self.next_data).to(device=self.device, non_blocking=True)
            if self.use_fp16:
                self.next_data = torch.from_numpy(self.next_data).to(dtype=torch.float32, non_blocking=True)

    def next(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        next_data = self.next_data
        self.preload()
        return next_data

    def get_recv_speed(self):
        return self.dataset.get_recv_speed()
