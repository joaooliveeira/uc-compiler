from mpi4py import MPI
import time

comm = MPI.COMM_WORLD

rank = comm.rank
size = comm.size
name = MPI.Get_processor_name()

if rank == 0:
    comm.send(rank, dest=1)
    comm.send(rank, dest=2)

    data = comm.recv(source=1)
    print('on node', rank, 'we received from:', data, "\n")

    data = comm.recv(source=2)
    print('on node', rank, 'we received:', data, "\n")

elif rank == 1:
    data = comm.recv(source=0)
    print('on node', rank, 'we received:', data, "\n")

    comm.send(rank, dest=0)
    comm.send(rank, dest=2)

    data = comm.recv(source=2)
    print('on node', rank, 'we received:', data, "\n")


elif rank == 2:
    data = comm.recv(source=0)
    print('on node', rank, 'we received:', data, "\n")

    comm.send(rank, dest=0)
    comm.send(rank, dest=1)

    data = comm.recv(source=1)
    print('on node', rank, 'we received:', data, "\n")

