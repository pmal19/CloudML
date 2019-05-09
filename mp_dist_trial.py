import os
import socket
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size) # or 'nccl'

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)
    
    
def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, world_size):
    setup(rank, world_size)
    print("mp rank - ", rank)
    hostname = socket.gethostname()
    runDistCollectives(rank, world_size, hostname)
    cleanup()
    
    
def runDistCollectives(rank, world_size, hostname):
    print("I am {} of {} in {}".format(rank, world_size, hostname))
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])
    
    
def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    

if __name__ == "__main__":
    run_demo(demo_basic, 2)