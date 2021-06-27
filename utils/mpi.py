from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()

def send(data, dest=0):
    comm.Send(data, dest=dest)

def receive(source, shape):
    data = np.empty(shape)
    comm.Recv(data, source=source)
    return data

def broadcast(data, root=0):
    comm.Bcast(data, root=root)
    # return data

def scatter(data, shape=None, root=0):
    buf = np.empty(data[0].shape if shape is None else shape)
    comm.Scatter(data, buf, root=0)
    return buf

def gather(data, root=0):
    rnk = comm.Get_rank()
    if rnk == root:
        buf = np.empty([size] + [d for d in data.shape])
    comm.Gather(data, buf, root=0)
    return buf

def mpi_print(*args):
    rnk = comm.Get_rank()
    print("[{}]".format(rnk), *args)

# TF-related MPI functions

def broadcast_model(model, root=0):
    rnk = comm.Get_rank()
    for layer in model.layers:
        layer_weights = []
        for weight in layer.get_weights():
            broadcast(weight)
            layer_weights.append(weight)
        if rnk > 0:
            layer.set_weights(layer_weights)

def average_gradients(grads, root=0):
    rnk = comm.Get_rank()
    res = np.zeros_like(grads)
    comm.Allreduce(grads, res, op=MPI.SUM)
    return res / size

