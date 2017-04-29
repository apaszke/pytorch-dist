.. distributed:: torch.

Modes
=====

There are two modes to control the way calculations are distributed: master-worker mode and process-group mode.
Each of them is designed to resemble interfaces and APIs familiar to PyTorch users.

The process-group mode API is made to look like the MPI API.
It is based on the idea that every machine in the network should queue jobs for itself.
Its interface is designed to give the programmer more flexibility than the master-worker mode which naturally makes its API a little bit more complex.

.. TODO: code examples

The master-worker mode on the other hand is dedicated to the users familiar with Nvidia CUDA.
The API is simple and makes it easy to start using it right away.
This model does not scale well due to the bottleneck in the master node which is responsible for all the planning job queuing in all the worker nodes.
Nevertheless, it is sufficient for networks of few machines.

.. TODO: code examples

Distributed computing
=====================

In order to configure connections for a program using THD it is required to set a couple of environmental variables.
Depending on the data channel implementation and selected mode it might be required to set a different subset of those variables.

Environmental variables
^^^^^^^^^^^^^^^^^^^^^^^
.. TODO: Explain the usage of the environmental variables.

Here is a list of the environmental variables which configure THD:

- `RANK`
  It is used to differentiate nodes within the network from each other.
  It should be unique and in range `0`-`$WORLD_SIZE-1`.

- `WORLD_SIZE`
  It is used to pass the information about the size of a network to nodes.

- `MASTER_PORT`
  This variable should be set to the number of a port which worker nodes use to establish connection with a master node.
  It needs to be set in the master node (worker nodes ignore it).

- `MASTER_ADDR`
  This variable should set to the address of a master node and the value of `MASTER_PORT` separated by a colon.
  For example `server15:29500` or `192.168.70.15:29500`.
  It needs to be set in worker nodes (the master node ignores it).

How to launch a distributed program?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's assume we have got a network of 3 machines called node0, node1 and node2.
Our PyTorch script is located in `~jim/project` and is called `training.py`.
The MPI examples were tested on the MPICH implementation -  specifically, the `mpirun` launcher differs between MPI implementations.

Process-group mode
------------------

MPI
~~~

::
    # On node0.
    mpirun -hosts node0,node1,node2 -n 3 python ~jim/project/training.py

The world size (number of nodes) has to be passed to `mpirun` using the `-n` option.
It is required by the MPI to how many processes it should spawn on the hosts provided with the `-hosts` option.

It is not required to set any THD-specific environmental variables when using MPI.

.. Warn the user that mpirun might differ between MPI implementations.
Remember that `mpirun` is not standardized by the MPI.
Even if you are not using the MPICH implementation it should be pretty straightforward to set everything up.

TCP & Gloo
~~~~~~~~~~

::
    # On node0:
    export WORLD_SIZE=3 RANK=0 MASTER_PORT=29500
    python ~jim/project/training.py

    # On node1:
    export WORLD_SIZE=3 RANK=1 MASTER_ADDR=node0:29500
    python ~jim/project/training.py

    # On node2:
    export WORLD_SIZE=3 RANK=2 MASTER_ADDR=node0:29500
    python ~jim/project/training.py

node0 acts as a master node here.

.. TODO: Make a distinction between the p-g mode and the m-w mode.

API
===
.. TODO: Document the API.

.. ............................................................................
.. General notes:
   TODO: Fix formatting.
   TODO: Focus on p-g mode.
