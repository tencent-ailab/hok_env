import os
import torch
import torch.distributed as dist


class NodeInfo(object):
    def __init__(self, rank=None, rank_size=None, local_rank=None, local_size=None) -> None:
        if rank is not None and rank_size is not None and local_rank is not None and local_size is not None:
            self.rank = rank
            self.rank_size = rank_size
            self.local_rank = local_rank
            self.local_size = local_size
        # mpirun
        if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
            self.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
            self.rank_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
            self.local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
            self.local_size = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])
            master_uri = "tcp://{ip}:{port}".format(ip=os.environ["MASTER_ADDR"], port=os.environ["MASTER_PORT"])
            if self.rank_size > 1:
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "mpi",
                    init_method=master_uri,
                    rank=self.rank,
                    world_size=self.rank_size
                )
        # torchrun
        elif "LOCAL_RANK" in os.environ:
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
            self.rank = int(dist.get_rank())
            self.rank_size = int(dist.get_world_size())
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.local_size = int(os.environ["LOCAL_WORLD_SIZE"])
        else:
            self.rank = 0
            self.rank_size = 1
            self.local_rank = 0
            self.local_size = 1

        self.is_chief_rank = self.rank == 0
