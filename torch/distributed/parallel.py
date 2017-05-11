import sys
import threading

import torch
from torch import nn
from torch.autograd import Variable
from torch._utils import _flatten_tensors, _unflatten_tensors
from torch.cuda.comm import broadcast_coalesced
from torch.cuda import nccl
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply

from . import _register_stream
from .collectives import all_reduce, broadcast, get_num_processes, get_rank, barrier

if sys.version_info[0] == 3:
    import queue
else:
    import Queue as queue


class DistributedDataParallel(nn.Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DistributedDataParallel, self).__init__()

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device

        # Sync params and buffers
        for p in self.module.state_dict().values():
            broadcast(p, 0)
            if get_rank() == 1:
                torch.cuda.synchronize() # TODO: fix this in Gloo
            barrier()

        if len(device_ids) > 1:
            # TODO: we don't need to replicate params in here. they're always going to
            # be broadcasted using larger blocks in broadcast_coalesce, so it might be
            # better to not pollute the caches with these small blocks
            self._module_copies = replicate(self.module, self.device_ids)
            self._module_copies[0] = self.module
            for module_copy in self._module_copies[1:]:
                for param, copy_param in zip(self.module.parameters(), module_copy.parameters()):
                    copy_param.detach_()
                    copy_param.requires_grad = param.requires_grad
        else:
            self._modules_copies = [self.module]

        # Split parameters into buckets that will coalesce reductions
        # TODO: different types need different buckets
        t = None
        for p in self.module.parameters():
            tp = type(p.data)
            if t is not None and t is not tp:
                raise ValueError("DistributedDataParallel requires all parameters' data to be of the same type")
            t = tp

        self.bucket_bytes_cap = 10 * 1024 * 1024  # 10 MB
        self.bucket_sizes = []
        self.bucket_map = {}
        bucket_bytes = self.bucket_bytes_cap  # to init the first bucket immediately
        for param_tuple in zip(*map(lambda m: m.parameters(), self._module_copies)):
            if bucket_bytes >= self.bucket_bytes_cap:
                self.bucket_sizes.append(0)
                bucket_bytes = 0
            self.bucket_sizes[-1] += 1
            for p in param_tuple:
                self.bucket_map[p] = len(self.bucket_sizes) - 1
            bucket_bytes += p.numel() * p.element_size()

        self.buckets = [[[] for _ in range(len(self.device_ids))] for _ in range(len(self.bucket_sizes))]
        self.bucket_events = [[None] * len(self.device_ids) for _ in range(len(self.bucket_sizes))]
        self.reduced = [False] * len(self.bucket_sizes)

        self._register_grad_hooks()

        self.dispatch_lock = threading.Lock()
        self._start_reduction_thread()

    def __getstate__(self):
        attrs = copy.copy(self.__dict__)
        del attrs['_grad_accs'], attrs['_reduction_thread']

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._register_grad_hooks()
        self._start_reduction_thread()

    def forward(self, *inputs, **kwargs):
        if len(self.device_ids) == 1:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        self._sync_params()
        outputs = self.parallel_apply(self._module_copies, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

    def train(self, mode=True):
        super(DistributedDataParallel, self).train(mode)
        for module in self._module_copies[1:]:
            module.train(mode)

    def _sync_params(self):
        params = [p.data for p in self.module.parameters()]
        result = broadcast_coalesced(params, self.device_ids, self.bucket_bytes_cap)
        for tensors, module in zip(result[1:], self._module_copies[1:]):
            for tensor, param in zip(tensors, module.parameters()):
                param.data.set_(tensor) # TODO: assign instead of set_ (to handle device changes)

        # cross-node buffer sync
        buffers = list(self.module._all_buffers())
        flat_buffers = _flatten_tensors(buffers)
        broadcast(flat_buffers, 0)
        for buf, synced in zip(buffers, _unflatten_tensors(flat_buffers, buffers)):
            buf.copy_(synced)

        # intra-node buffer sync
        result = broadcast_coalesced(buffers, self.device_ids, self.bucket_bytes_cap)
        for tensors, module in zip(result[1:], self._module_copies[1:]):
            for tensor, buf in zip(tensors, module._all_buffers()):
                buf.set_(tensor)

    def _register_grad_hooks(self):
        self._grad_accs = []  # need to keep them in scope
        for device_idx, module in enumerate(self._module_copies):
            for p in module.parameters():
                # TODO: no-op for these that don't require grad
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_param_hook(p, device_idx))
                self._grad_accs.append(grad_acc)

    def _make_param_hook(self, param, device_idx):
        bucket_idx = self.bucket_map[param]
        def dist_dp_hook(*unused):
            if not param.grad.volatile:
                raise RuntimeError("DistributedDataParallel only works with volatile gradients")
            bucket = self.buckets[bucket_idx][device_idx]
            bucket.append(param.grad.data)

            # We can flush these and save memory for replicas
            if device_idx > 0:
                param.grad = None
                param.data.set_()

            # Current device's bucket is full
            if len(bucket) == self.bucket_sizes[bucket_idx]:
                with torch.cuda.device(self.device_ids[device_idx]):
                    event = torch.cuda.Event()
                    event.record()
                with self.dispatch_lock:
                    self.bucket_events[bucket_idx][device_idx] = event
                    self._queue_reduction(bucket_idx)

        return dist_dp_hook

    def _queue_reduction(self, bucket_idx):
        while bucket_idx >= 0:
            dev_buckets = self.buckets[bucket_idx]
            dev_events = self.bucket_events[bucket_idx]

            # Check if it's ready
            if any(evt is None for evt in dev_events):
                return
            # We always reduce the buckets from larger indices to smaller, so we
            # have to check that all buckets to the right have already queued reductions.
            is_last = bucket_idx == len(self.buckets) - 1
            if not is_last and not self.reduced[bucket_idx + 1]:
                return

            # Queue the reduction and make sure backward waits for it
            event = threading.Event()
            self.reduction_queue.put((dev_buckets, dev_events, event, bucket_idx == 0))
            Variable._execution_engine.queue_callback(lambda: event.wait())

            # Reset bucket state
            self.buckets[bucket_idx] = [[] for _ in range(len(self.device_ids))]
            self.bucket_events[bucket_idx] = [None] * len(self.device_ids)
            self.reduced[bucket_idx] = True
            if bucket_idx == 0:
                self.reduced = [False] * len(self.bucket_sizes)

            # Try previous bucket
            bucket_idx -= 1

    def _start_reduction_thread(self):
        self.reduction_queue = queue.Queue()
        default_streams = []
        reduction_streams = []
        for dev_id in self.device_ids:
            with torch.cuda.device(dev_id):
                # TODO: don't assume we're on the default stream
                default_streams.append(torch.cuda.current_stream())
                reduction_streams.append(torch.cuda.Stream())
        _register_stream(reduction_streams[0])

        self._reduction_thread = threading.Thread(
            target=self._reduction_thread_fn,
            args=(self.reduction_queue, self.device_ids, reduction_streams, default_streams))
        self._reduction_thread.start()

    @staticmethod
    def _reduction_thread_fn(queue, device_ids, reduction_streams, default_streams):
        def _process_batch():
            dev_grad_batch, dev_events, job_event, is_last = queue.get()
            dev_coalesced = []
            # Coalesce the tensors on all devices and start a local reduction
            for dev_id, grad_batch, event, stream in zip(device_ids, dev_grad_batch, dev_events, reduction_streams):
                with torch.cuda.device(dev_id), torch.cuda.stream(stream):
                    stream.wait_event(event)
                    coalesced = _flatten_tensors(grad_batch)
                    dev_coalesced.append(coalesced)
            nccl.reduce(dev_coalesced, root=device_ids[0], streams=reduction_streams)

            # From now on we're only going to work on the first device (from device_ids)
            grad_batch = dev_grad_batch[0]
            coalesced = dev_coalesced[0]
            reduce_stream = reduction_streams[0]
            with torch.cuda.stream(reduce_stream):
                coalesced /= get_num_processes()
                all_reduce(coalesced)
                for grad, reduced in zip(grad_batch, _unflatten_tensors(coalesced, grad_batch)):
                    grad.copy_(reduced)
                barrier() # TODO: this should be unnecessary once the bugs in Gloo are fixed

            # Insert default stream sync after queuing kernels from the last bucket
            if is_last:
                def sync_streams():
                    for dev_id, default_stream, reduce_stream in zip(device_ids, default_streams, reduction_streams):
                        with torch.cuda.device(dev_id):
                            default_stream.wait_stream(reduce_stream)
                Variable._execution_engine.queue_callback(sync_streams)
            job_event.set()

        with torch.cuda.device(device_ids[0]):
            while True:
                _process_batch()  # just to have a clear scope
