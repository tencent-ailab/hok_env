import torch


class StepContext:
    def __init__(self, rank, local_rank, gpu_nums, ip, batch_size):
        self.rank = rank
        self.local_rank = local_rank
        self.ip = ip
        self.batch_size = batch_size
        self.gpu_nums = gpu_nums
        self.is_chief_rank = self.rank == 0
        self.total_loss = None
        self.info_list = None
        self.step = None
        self.sample_recv_speed = None
        self.sample_consume_speed = None
        self.train_has_inf_nan = False
        self.grad_has_inf_nan = False

    def set_forward_info(self, total_loss, info_list):
        # deep copy gpu tensor to cpu in case that method optm.step() changes parameters
        self.total_loss = total_loss.to("cpu", copy=True) if isinstance(total_loss, torch.Tensor) else total_loss
        self.info_list = [(item.to("cpu", copy=True) if isinstance(item, torch.Tensor) else item) for item in info_list]

    def check_has_inf_nan(self, total_loss, params):
        self.train_has_inf_nan = False
        self.grad_has_inf_nan = False
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            self.train_has_inf_nan = True
        for param in params:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                self.grad_has_inf_nan = True
                break

    def set_other_info(self, step, sample_recv_speed, sample_consume_speed):
        self.step = step
        self.sample_recv_speed = sample_recv_speed
        self.sample_consume_speed = sample_consume_speed

    def decode(self):
        return {
            "step": self.step,
            "batch_size": self.batch_size,
            "gpu_nums": self.gpu_nums,
            "sample_recv_speed": self.sample_recv_speed,
            "sample_consume_speed": self.sample_consume_speed,
            "total_loss": self.total_loss,
            "info_list": self.info_list,
            "train_has_inf_nan": self.train_has_inf_nan,
            "grad_has_inf_nan": self.grad_has_inf_nan,
        }