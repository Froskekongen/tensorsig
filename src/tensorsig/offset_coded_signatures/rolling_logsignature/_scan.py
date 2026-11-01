"""Grade-by-grade affine doubling scan (O(N log N) combines)."""

import torch

from ._compile import Transport, pascal_transport


class _AffineScan(torch.autograd.Function):
    """Linear scan whose reverse uses the same primitive at every derivative order."""

    @staticmethod
    def forward(ctx, forcing, initial, powers):
        ctx.powers = powers
        ctx.initial_shape = initial.shape
        return affine_scan(forcing, initial, powers)

    @staticmethod
    def backward(ctx, cotangent):
        if not cotangent.shape[-2]:
            return cotangent, cotangent.new_zeros(ctx.initial_shape), None
        transpose = tuple(matrix.T for matrix in ctx.powers)
        reverse = _AffineScan.apply(
            cotangent.flip(-2), cotangent.new_zeros(ctx.initial_shape), transpose
        ).flip(-2)
        return reverse, reverse[..., 0, :] @ transpose[0], None


def adjoint_affine_scan(forcing, initial, powers):
    """Inclusive affine scan without saving its doubling levels for autograd."""
    if any(matrix.requires_grad for matrix in powers):
        raise ValueError("Affine scan transport is fixed, not differentiable metadata.")
    return _AffineScan.apply(forcing, initial, tuple(powers))


class _GradeScan(torch.autograd.Function):
    """Save final trajectories; recompute only grade forcing in the reverse pass."""

    @staticmethod
    def forward(ctx, initial, outgoing, incoming, stages):
        trajectory = initial.new_zeros(
            initial.shape[0], outgoing.shape[1] + 1, initial.shape[-1]
        )
        for indices, program, powers in stages:
            forcing = program._forward(trajectory[:, :-1], outgoing, incoming)
            values = affine_scan(forcing, initial[..., indices], powers)
            values = torch.cat((initial[:, None, indices], values), dim=1)
            trajectory = trajectory.index_copy(-1, indices, values)
        ctx.stages = stages
        # Saving the returned trajectory preserves its custom grad_fn for double backward.
        ctx.save_for_backward(trajectory, outgoing, incoming)
        return trajectory

    @staticmethod
    def backward(ctx, cotangent):
        trajectory, outgoing, incoming = ctx.saved_tensors
        initial_bar = torch.zeros_like(trajectory[:, 0])
        outgoing_bar, incoming_bar = (
            torch.zeros_like(outgoing),
            torch.zeros_like(incoming),
        )
        for indices, program, powers in reversed(ctx.stages):
            # The last-to-first recurrence includes the initial state's cotangent.
            transpose = tuple(matrix.T for matrix in powers)
            grade_bar = cotangent[..., indices]
            adjoints = adjoint_affine_scan(
                grade_bar.flip(1), torch.zeros_like(grade_bar[:, 0]), transpose
            ).flip(1)
            initial_bar = initial_bar.index_copy(-1, indices, adjoints[:, 0])
            lower_bar, out_bar, in_bar = program.vjp(
                (trajectory[:, :-1], outgoing, incoming), adjoints[:, 1:]
            )
            cotangent = cotangent + torch.cat(
                (lower_bar, torch.zeros_like(lower_bar[:, :1])), dim=1
            )
            outgoing_bar = outgoing_bar + out_bar
            incoming_bar = incoming_bar + in_bar
        return initial_bar, outgoing_bar, incoming_bar, None


def optimized_scan_interval(owner, increments, starts, length, backend):
    """Joint forcing and its explicit, differentiable grade-reversed adjoint."""
    # Initialization has the same independent-window contract as the FIFO route.
    if backend in {"joint", "joint_fused"}:
        append = owner._general_program(backend == "joint_fused", right_degree_one=True)
    else:
        append = owner._endpoints(backend)[0]
    batch, lanes, dimension = increments.shape[0], len(starts), owner.plan.dimension
    initial = increments.new_zeros(batch, lanes, dimension)
    for r in range(owner.window):
        initial = append(initial, increments[:, starts + r] @ owner.embeddings[r])
    if length == 1 or not dimension:
        return initial.unsqueeze(2).expand(batch, lanes, length, dimension)
    count = length - 1
    positions = starts[:, None] + torch.arange(count, device=increments.device)
    positions = positions.clamp_max(increments.shape[1] - owner.window - 1)
    outgoing = (increments[:, positions] @ owner.embeddings[0]).flatten(0, 1)
    incoming = (increments[:, positions + owner.window] @ owner.embeddings[-1]).flatten(
        0, 1
    )
    # Backward scans N+1 cotangents, so compile enough powers for that length too.
    key = (length, backend == "joint_fused")
    if key not in owner._scan_stages:
        transports = owner._scan_powers(length)
        stages = []
        for grade in range(1, owner.plan.max_depth + 1):
            indices = torch.where(owner.plan.basis_depths == grade)[0]
            if not len(indices):
                continue
            matrices = tuple(
                next(
                    matrix
                    for ids, matrix in power.blocks
                    if int(owner.plan.basis_depths[ids[0]]) == grade
                )
                for power in transports
            )
            stages.append((indices, owner._joint_program(grade, key[1]), matrices))
        owner._scan_stages[key] = tuple(stages)
    result = _GradeScan.apply(
        initial.flatten(0, 1), outgoing, incoming, owner._scan_stages[key]
    )
    return result.reshape(batch, lanes, length, dimension)


def affine_scan(forcing, initial, powers):
    """Inclusive states for x[t+1]=T x[t]+b[t], with x[0]=initial.

    powers[k] is the row-vector matrix T**(2**k). Autograd transposes this
    functional DAG; it retains scan levels instead of using a custom adjoint.
    """
    count = forcing.shape[-2]
    if not count:
        return forcing
    state = torch.cat(
        (forcing[:, :1] + (initial @ powers[0]).unsqueeze(1), forcing[:, 1:]), dim=1
    )
    step = 1
    for matrix in powers:
        if step >= count:
            break
        state = torch.cat(
            (state[:, :step], state[:, step:] + state[:, :-step] @ matrix), dim=1
        )
        step *= 2
    return state


def prepare_powers(owner, count):
    needed = max(1, (max(1, count) - 1).bit_length())
    while len(owner._powers) < needed:
        shift = -(1 << len(owner._powers))
        owner._powers.append(Transport(owner.plan, pascal_transport(owner.plan, shift)))
    return owner._powers[:needed]
