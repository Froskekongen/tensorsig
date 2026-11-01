"""Measured allocator/tape/operator metrics; allocation is not DRAM traffic."""

from contextlib import nullcontext

import torch


def profile_call(
    evaluate,
    inputs,
    weights,
    direction=None,
    *,
    derivative_order=None,
    trace_operators=False,
):
    """Inspect saved storage; opt into memory-intensive operator event tracing."""
    if derivative_order is None:
        derivative_order = 2 if direction is not None else 1
    if derivative_order not in (0, 1, 2) or (
        derivative_order == 2 and direction is None
    ):
        raise ValueError("Profile order must be 0/1/2, with a direction for order 2.")
    saved = []
    storages = {}

    def pack(tensor):
        saved.append(tensor.numel() * tensor.element_size())
        if tensor.layout == torch.sparse_coo:
            components = (tensor._indices(), tensor._values())
        elif tensor.layout in (torch.sparse_csr, torch.sparse_bsr):
            components = (tensor.crow_indices(), tensor.col_indices(), tensor.values())
        elif tensor.layout in (torch.sparse_csc, torch.sparse_bsc):
            components = (tensor.ccol_indices(), tensor.row_indices(), tensor.values())
        else:
            components = (tensor,)
        for component in components:
            storage = component.untyped_storage()
            storages[(str(component.device), storage.data_ptr())] = storage.nbytes()
        return tensor

    activities = [torch.profiler.ProfilerActivity.CPU]
    if inputs.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace = (
        torch.profiler.profile(activities=activities, profile_memory=True)
        if trace_operators
        else nullcontext()
    )
    with trace as prof:
        if derivative_order == 0:
            with torch.no_grad():
                result = evaluate(inputs)
        else:
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                result = evaluate(inputs)
                if derivative_order == 2:
                    loss = (result * weights + result.square() / 2).sum()
                    gradient = torch.autograd.grad(
                        loss + inputs.square().sum() * 0, inputs, create_graph=True
                    )[0]
            if derivative_order == 1:
                torch.autograd.grad((result * weights).sum(), inputs)
            else:
                torch.autograd.grad((gradient * direction).sum(), inputs)
    events = prof.events() if trace_operators else None
    cuda_events = (
        [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
        if events is not None
        else None
    )
    return {
        "derivative_order": derivative_order,
        "operator_trace_enabled": trace_operators,
        "scope": (
            "one inference call without autograd",
            "one forward and VJP after warmup; saved tensors cover forward only",
            "one nonlinear-loss HVP; saved tensors cover forward and first-gradient construction",
        )[derivative_order],
        "saved_tensor_references": len(saved),
        "saved_tensor_logical_bytes": sum(saved),
        "saved_unique_storage_bytes": sum(storages.values()),
        "cpu_operator_events": sum(e.name.startswith("aten::") for e in events)
        if events is not None
        else None,
        "cpu_positive_self_allocation_bytes": sum(
            max(0, e.self_cpu_memory_usage) for e in events
        )
        if events is not None
        else None,
        "cuda_device_events": len(cuda_events)
        if inputs.device.type == "cuda" and cuda_events is not None
        else None,
        "cuda_kernel_events": sum(
            not e.name.startswith(("Memcpy", "Memset")) for e in cuda_events
        )
        if inputs.device.type == "cuda" and cuda_events is not None
        else None,
        "gpu_launch_count": None
        if inputs.device.type != "cuda" or events is None
        else sum(
            "LaunchKernel" in e.name or "LaunchCooperativeKernel" in e.name
            for e in events
        ),
        "dram_bytes": None,
        "limitations": "Operator tracing is optional and retains event records that can materially "
        "increase process RSS; disable it for deployment memory measurements. "
        "Event-derived metrics are unavailable when tracing is disabled. "
        "CPU allocation bytes are cumulative, not peak or traffic. "
        "Saved logical bytes count repeated references; unique storage includes "
        "whole aliased buffers and constants. Sparse logical bytes use dense shape; "
        "unique storage counts sparse index/value components. MPS launches/traffic unavailable "
        "from this profiler; CUDA kernel names exclude copies/memsets heuristically.",
    }
