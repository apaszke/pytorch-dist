#!/usr/bin/env bash
set -e

PYCMD=${PYCMD:="python"}
if [ "$1" == "coverage" ];
then
    coverage erase
    PYCMD="coverage run --parallel-mode --source torch "
    echo "coverage flag found. Setting python command to: \"$PYCMD\""
fi

pushd "$(dirname "$0")"

echo "Running torch tests"
$PYCMD test_torch.py

echo "Running autograd tests"
$PYCMD test_autograd.py

echo "Running nn tests"
$PYCMD test_nn.py

echo "Running legacy nn tests"
$PYCMD test_legacy_nn.py

echo "Running optim tests"
$PYCMD test_optim.py

echo "Running multiprocessing tests"
$PYCMD test_multiprocessing.py
MULTIPROCESSING_METHOD=spawn $PYCMD test_multiprocessing.py
MULTIPROCESSING_METHOD=forkserver $PYCMD test_multiprocessing.py

echo "Running util tests"
$PYCMD test_utils.py

echo "Running dataloader tests"
$PYCMD test_dataloader.py

if which nvcc >/dev/null 2>&1
then
    echo "Running cuda tests"
    $PYCMD test_cuda.py

    echo "Running NCCL tests"
    $PYCMD test_nccl.py
else
    echo "nvcc not found in PATH, skipping CUDA tests"
fi

################################################################################
distributed_set_up() {
    export TEMP_DIR="$(mktemp -d)"
    rm -rf "$TEMP_DIR/"*
    mkdir "$TEMP_DIR/barrier"
    mkdir "$TEMP_DIR/test_dir"
}

distributed_tear_down() {
    rm -rf "$TEMP_DIR"
}

trap distributed_tear_down EXIT SIGHUP SIGINT SIGTERM

echo "Running distributed tests for the TCP backend"
distributed_set_up
BACKEND=tcp WORLD_SIZE=3 $PYCMD ./test_distributed.py
distributed_tear_down

echo "Running distributed tests for the MPI backend"
distributed_set_up
BACKEND=mpi mpiexec -n 3 $PYCMD ./test_distributed.py
distributed_tear_down
################################################################################

if [ "$1" == "coverage" ];
then
    coverage combine
    coverage html
fi

popd
