#!/usr/bin/env python3
"""Exact PIT2 common-source coordinates, image ranks, and replayable decoders.

Port of the retained ``evaluate_graph_block_coordinates.py`` proof evaluator.
The source program, normalization, prescribed dimensions, and replay contract
are preserved. This is an offline rational compiler, not a floating GPU chart.
The independent bounded controls live in the test suite; the standalone
``scripts/evaluate_graph_block_coordinates.py`` command evaluates and replays JSON.
"""

import hashlib
import json
import re
from collections import defaultdict, deque
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from math import comb, factorial, prod
from pathlib import Path

from ._graph_common import ResourceLimit

# Local serialization limits; importing a library must not change Python's
# process-wide integer parser/formatter policy.
DECIMAL_DIGIT_LIMIT = 200_000

TARGET_DIGEST = "04a205b1199e154f1973ba59ea8d0a937af8c6fbc3084aa7f63c9bbc929df852"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid {name}")
    return value


def _integer_text(value):
    # 3.322 is a conservative upper bound for log2(10). Check before conversion.
    budget(
        abs(value).bit_length(), DECIMAL_DIGIT_LIMIT * 3322 // 1000 + 1, "integer bits"
    )
    text = str(Decimal(value))
    budget(len(text.lstrip("-")), DECIMAL_DIGIT_LIMIT, "decimal digits")
    return text


def _decimal_integer(text):
    text = text.strip()
    require(
        re.fullmatch(r"[+-]?[0-9]+(?:_[0-9]+)*", text) is not None,
        "invalid integer coefficient",
    )
    digits = text.lstrip("+-").replace("_", "")
    budget(len(digits), DECIMAL_DIGIT_LIMIT, "decimal digits")
    return int(Decimal(text.replace("_", "")))


def rational(value):
    """Parse an exact coefficient with a local 200,000-digit ceiling.

    Use this for serialized coordinates that exceed Python's process-wide
    integer-string limit; it never changes that global setting.
    """
    require(
        type(value) in (int, str), "rational coefficients must be integers or strings"
    )
    if type(value) is int:
        budget(
            abs(value).bit_length(),
            DECIMAL_DIGIT_LIMIT * 3322 // 1000 + 1,
            "integer bits",
        )
        return Fraction(value)
    try:
        if "/" in value:
            numerator, denominator = value.split("/")
            return Fraction(_decimal_integer(numerator), _decimal_integer(denominator))
        # Decimal(int/string) is exact and independent of the decimal context.
        # Check the expanded exponent before Fraction allocates a huge power.
        budget(len(value), DECIMAL_DIGIT_LIMIT + 32, "decimal input characters")
        digits_pattern = r"[0-9]+(?:_[0-9]+)*"
        decimal_pattern = rf"[+-]?(?:{digits_pattern}(?:\.(?:{digits_pattern})?)?|\.{digits_pattern})(?:[eE][+-]?{digits_pattern})?"
        require(
            re.fullmatch(decimal_pattern, value.strip()) is not None,
            "invalid decimal coefficient",
        )
        decimal = Decimal(value)
        require(decimal.is_finite(), "nonfinite rational coefficient")
        if not decimal:
            return Fraction()
        _, digits, exponent = decimal.as_tuple()
        budget(
            max(len(digits) + exponent, 1 - exponent),
            DECIMAL_DIGIT_LIMIT,
            "decimal digits",
        )
        return Fraction(decimal)
    except ResourceLimit:
        raise
    except (ValueError, ZeroDivisionError, InvalidOperation, OverflowError) as exc:
        raise ValueError("invalid rational coefficient") from exc


def _rational_text(value):
    value = Fraction(value)
    numerator = _integer_text(value.numerator)
    return (
        numerator
        if value.denominator == 1
        else numerator + "/" + _integer_text(value.denominator)
    )


def _canonical_json(value):
    """Canonical JSON with locally bounded integers, including large literals."""
    if type(value) is int:
        return _integer_text(value)
    if isinstance(value, dict):
        require(
            all(isinstance(key, str) for key in value),
            "JSON object keys must be strings",
        )
        return (
            "{"
            + ",".join(
                json.dumps(key) + ":" + _canonical_json(value[key])
                for key in sorted(value)
            )
            + "}"
        )
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical_json(item) for item in value) + "]"
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def check_fields(obj, allowed):
    require(isinstance(obj, dict), "expected an object")
    require(not set(obj) - set(allowed), "unknown input fields")


def budget(value, limit, label):
    if value > limit:
        raise ResourceLimit(f"{label} exceeds the configured resource limit")


def compositions(total, width):
    if width == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for tail in compositions(total - first, width - 1):
                yield (first,) + tail


def graph_program(spec, limits):
    """Compile [z^alpha] sum_h (-1)^(h+1) binom(m,h) g^h/h, then scale by alpha!."""
    check_fields(spec, {"kind", "q", "channels", "edges", "occupancy", "cliques"})
    q = tuple(integer(x, "block size", 1) for x in spec["q"])
    require(q, "empty composition")
    C = integer(spec["channels"], "channel count", 1)
    s, m = len(q), sum(q)
    budget(s * C, limits["vertices"], "type count")
    edges = set()
    for edge in spec.get("edges", []):
        require(isinstance(edge, list) and len(edge) == 2, "invalid channel edge")
        a, b = (integer(x, "channel index") for x in edge)
        require(a < C and b < C and a != b, "invalid channel edge")
        pair = tuple(sorted((a, b)))
        require(pair not in edges, "duplicate channel edge")
        edges.add(pair)
    alpha = spec.get("occupancy")
    if alpha is not None:
        require(isinstance(alpha, list) and len(alpha) == s, "invalid occupancy rows")
        for i, row in enumerate(alpha):
            require(
                isinstance(row, list) and len(row) == C, "invalid occupancy columns"
            )
            require(
                all(type(x) is int and x >= 0 for x in row) and sum(row) == q[i],
                "invalid occupancy row sum",
            )
        alpha = tuple(x for row in alpha for x in row)
    else:
        budget(
            prod(comb(x + C - 1, C - 1) for x in q), limits["outputs"], "source outputs"
        )
    types = tuple(
        (i, c) for i in range(s) for c in range(C) if alpha is None or alpha[i * C + c]
    )
    caps = tuple(q[i] if alpha is None else alpha[i * C + c] for i, c in types)
    if "cliques" in spec:
        raw_cliques = spec["cliques"]
        require(isinstance(raw_cliques, list) and raw_cliques, "empty clique cover")
        cliques = []
        for raw in raw_cliques:
            require(isinstance(raw, list) and raw, "invalid clique")
            clique = tuple(integer(c, "clique channel") for c in raw)
            require(
                len(set(clique)) == len(clique) and all(c < C for c in clique),
                "invalid clique channels",
            )
            require(
                all(
                    tuple(sorted((a, b))) in edges
                    for a in clique
                    for b in clique
                    if a < b
                ),
                "cover contains a nonclique",
            )
            cliques.append(tuple(sorted(clique)))
        require(
            set().union(*(set(c) for c in cliques)) == set(range(C)),
            "cover misses a channel",
        )
        require(
            all(any(a in c and b in c for c in cliques) for a, b in edges),
            "cover misses a dependence edge",
        )
    elif len(edges) == C * (C - 1) // 2:
        cliques = [tuple(range(C))]
    else:
        covered = {c for e in edges for c in e}
        cliques = sorted(edges) + [(c,) for c in range(C) if c not in covered]
    factors = [tuple(t for t in types if t[1] in clique) for clique in cliques]
    factors = [factor for factor in factors if factor]
    type_letters = {
        t: [(j, factor.index(t) + 1) for j, factor in enumerate(factors) if t in factor]
        for t in types
    }
    rows = [tuple(k for k, t in enumerate(types) if t[0] == i) for i in range(s)]
    nodes, raw_edges, terminals = 1, [], {}

    def node():
        nonlocal nodes
        nodes += 1
        budget(nodes, limits["vertices"], "construction vertices")
        return nodes - 1

    zero = (0,) * len(types)
    for h in range(1, m + 1):
        states, queue = {}, deque()

        def state(key):
            if key not in states:
                states[key] = node()
                queue.append(key)
            return states[key]

        root = state((0, 0, zero))
        raw_edges.append((0, root, None, Fraction((-1) ** (h + 1) * comb(m, h), h)))
        while queue:
            segment, local, beta = key = queue.popleft()
            u = states[key]
            if segment == h * s:
                if all(sum(beta[k] for k in rows[i]) == q[i] for i in range(s)):
                    full = tuple(
                        beta[types.index((i, c))] if (i, c) in types else 0
                        for i in range(s)
                        for c in range(C)
                    )
                    if full not in terminals:
                        budget(len(terminals) + 1, limits["outputs"], "source outputs")
                        terminals[full] = node()
                    raw_edges.append((u, terminals[full], None, Fraction(1)))
                continue
            i = segment % s
            raw_edges.append((u, state((segment + 1, 0, beta)), None, Fraction(1)))
            if sum(beta[k] for k in rows[i]) < q[i]:
                for k in rows[i]:
                    if beta[k] < caps[k]:
                        updated = list(beta)
                        updated[k] += 1
                        v = state((segment, local + 1, tuple(updated)))
                        raw_edges.append((u, v, types[k], Fraction(1, local + 1)))
    # Remove paths that cannot reach a selected full-occupancy output.
    predecessors = defaultdict(list)
    for u, v, _, _ in raw_edges:
        predecessors[v].append(u)
    live, stack = set(terminals.values()), list(terminals.values())
    while stack:
        for u in predecessors[stack.pop()]:
            if u not in live:
                live.add(u)
                stack.append(u)
    require(0 in live, "source has no accepting path")
    old_to_new = {v: i for i, v in enumerate(sorted(live))}
    n = len(live)
    linear = []
    for u, v, label, coefficient in raw_edges:
        if u not in live or v not in live:
            continue
        u, v = old_to_new[u], old_to_new[v]
        letters = [None] if label is None else type_letters[label]
        for pos, letter in enumerate(letters):
            endpoint = v if pos == len(letters) - 1 else n
            if endpoint == n:
                n += 1
                budget(n, limits["vertices"], "linearized vertices")
            linear.append(
                [u, endpoint, letter, _rational_text(coefficient if pos == 0 else 1)]
            )
            u = endpoint
    outputs = []
    for full, endpoint in sorted(terminals.items()):
        weights = ["0"] * n
        weights[old_to_new[endpoint]] = _integer_text(prod(factorial(x) for x in full))
        outputs.append({"name": list(full), "weights": weights})
    start = ["0"] * n
    start[old_to_new[0]] = "1"
    return {
        "kind": "tensor-abp",
        "vertices": n,
        "alphabet_sizes": [len(f) for f in factors],
        "degree_bound": m,
        "edges": linear,
        "start": start,
        "outputs": outputs,
    }


def validate_program(spec, limits):
    check_fields(
        spec,
        {
            "kind",
            "vertices",
            "alphabet_sizes",
            "degree_bound",
            "edges",
            "start",
            "outputs",
        },
    )
    N = integer(spec["vertices"], "vertex count", 1)
    budget(N, limits["vertices"], "vertices")
    sizes = tuple(integer(n, "alphabet size", 1) for n in spec["alphabet_sizes"])
    require(sizes, "empty factor list")
    m = integer(spec["degree_bound"], "degree bound")
    require(
        isinstance(spec["start"], list) and len(spec["start"]) == N,
        "invalid start vector",
    )
    start = list(map(rational, spec["start"]))
    require(isinstance(spec["outputs"], list) and spec["outputs"], "empty outputs")
    budget(len(spec["outputs"]), limits["outputs"], "outputs")
    outputs, names = [], set()
    for output in spec["outputs"]:
        check_fields(output, {"name", "weights"})
        name = _canonical_json(output["name"])
        require(name not in names, "duplicate output name")
        names.add(name)
        require(
            isinstance(output["weights"], list) and len(output["weights"]) == N,
            "invalid output vector",
        )
        outputs.append((output["name"], list(map(rational, output["weights"]))))
    adjacency, indegree = [[] for _ in range(N)], [0] * N
    for edge in spec["edges"]:
        require(isinstance(edge, list) and len(edge) == 4, "invalid ABP edge")
        u, v, letter, coefficient = edge
        integer(u, "edge source")
        integer(v, "edge target")
        require(u < N and v < N, "edge endpoint out of range")
        if letter is not None:
            require(
                isinstance(letter, (list, tuple)) and len(letter) == 2, "invalid letter"
            )
            j, i = letter
            integer(j, "factor index")
            integer(i, "letter index", 1)
            require(j < len(sizes) and i <= sizes[j], "letter out of range")
            letter = (j, i)
        coefficient = rational(coefficient)
        adjacency[u].append((v, letter, coefficient))
        indegree[v] += 1
    queue = deque(v for v in range(N) if not indegree[v])
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v, _, _ in adjacency[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    require(len(order) == N, "program is cyclic")
    # This executable accepts a verifiable pathwise bound, even if cancellation could lower degree.
    for j in range(len(sizes)):
        degree = [0 if x else -1 for x in start]
        for u in order:
            if degree[u] >= 0:
                for v, letter, coefficient in adjacency[u]:
                    if coefficient:
                        degree[v] = max(
                            degree[v],
                            degree[u] + int(letter is not None and letter[0] == j),
                        )
        require(
            all(degree[v] <= m for _, out in outputs for v in range(N) if out[v]),
            "pathwise factor degree exceeds bound",
        )
    return N, sizes, m, start, outputs, adjacency, order


def dimensions(N, sizes, m, limit):
    D, state_bounds, dims = 1, [], []
    for j, n in enumerate(sizes):
        S = N * D * (m + 1) ** (len(sizes) - j - 1)
        d = (n - 1) * comb(S, 2) + S
        D *= d
        budget(D, limit, "coordinate dimension")
        state_bounds.append(S)
        dims.append(d)
    return state_bounds, tuple(dims), D


def letter_action(vector, dims, letter):
    j, i = letter
    stride, d = prod(dims[j + 1 :]), dims[j]
    result = [Fraction(0)] * len(vector)
    for block in range(0, len(vector), d * stride):
        for offset in range(stride):
            c = Fraction(0)
            for u in range(1, d):
                c = (vector[block + (u - 1) * stride + offset] - c) / i
                result[block + u * stride + offset] = c / u
    return result


def coordinates(program, dims, output):
    N, _, _, start, _, adjacency, order = program
    D = prod(dims)
    values = [[Fraction(0)] * D for _ in range(N)]
    for u in reversed(order):
        values[u][0] += output[u]
        for v, letter, coefficient in adjacency[u]:
            if coefficient:
                image = (
                    values[v]
                    if letter is None
                    else letter_action(values[v], dims, letter)
                )
                for k, value in enumerate(image):
                    if value:
                        values[u][k] += coefficient * value
    return [
        sum((start[v] * values[v][k] for v in range(N) if start[v]), Fraction(0))
        for k in range(D)
    ]


def decoder(columns):
    """Select original independent columns and pivot rows; retain a square inverse."""
    reduced, chosen, pivots = [], [], []
    for j, column in enumerate(columns):
        v = list(column)
        for pivot, row in zip(pivots, reduced):
            coefficient = v[pivot]
            if coefficient:
                v = [x - coefficient * y for x, y in zip(v, row)]
        pivot = next((i for i, x in enumerate(v) if x), None)
        if pivot is not None:
            scale = v[pivot]
            reduced.append([x / scale for x in v])
            chosen.append(j)
            pivots.append(pivot)
    r = len(chosen)
    aug = [
        [columns[j][i] for j in chosen] + [Fraction(k == h) for k in range(r)]
        for h, i in enumerate(pivots)
    ]
    for k in range(r):
        p = next(h for h in range(k, r) if aug[h][k])
        aug[k], aug[p] = aug[p], aug[k]
        scale = aug[k][k]
        aug[k] = [x / scale for x in aug[k]]
        for h in range(r):
            if h != k:
                scale = aug[h][k]
                aug[h] = [x - scale * y for x, y in zip(aug[h], aug[k])]
    return {
        "basis_columns": chosen,
        "pivot_rows": pivots,
        "inverse": [[_rational_text(x) for x in row[r:]] for row in aug],
    }


def decode_image(columns, data, vector):
    require(len(vector) == len(columns[0]), "wrong coordinate dimension")
    coefficients = [
        sum(
            (rational(a) * vector[i] for a, i in zip(row, data["pivot_rows"])),
            Fraction(0),
        )
        for row in data["inverse"]
    ]
    reconstructed = [
        sum(
            (x * columns[j][k] for x, j in zip(coefficients, data["basis_columns"])),
            Fraction(0),
        )
        for k in range(len(vector))
    ]
    require(reconstructed == vector, "coordinate vector is outside the source image")
    return coefficients


def evaluate(spec, limits=None):
    limits = (
        {"vertices": 5000, "dimension": 20000, "outputs": 128}
        if limits is None
        else dict(limits)
    )
    require(
        set(limits) == {"vertices", "dimension", "outputs"}, "invalid resource limits"
    )
    require(
        all(type(x) is int and x > 0 for x in limits.values()),
        "limits must be positive integers",
    )
    require(isinstance(spec, dict), "expected input object")
    require(spec.get("kind") in {"graph-block", "tensor-abp"}, "unknown input kind")
    raw = graph_program(spec, limits) if spec["kind"] == "graph-block" else spec
    program = validate_program(raw, limits)
    N, sizes, m, _, outputs, _, _ = program
    states, dims, D = dimensions(N, sizes, m, limits["dimension"])
    columns = [coordinates(program, dims, out) for _, out in outputs]
    data = decoder(columns)
    return {
        "schema": 1,
        "method": "PIT2-tensor-first-column-v1",
        "input_sha256": digest(spec),
        "program_sha256": digest(raw),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "target_digest": TARGET_DIGEST,
        "vertices": N,
        "alphabet_sizes": list(sizes),
        "degree_bound": m,
        "slice_states": states,
        "factor_dimensions": list(dims),
        "dimension": D,
        "columns": [
            {
                "name": name,
                "coordinates": [[i, _rational_text(x)] for i, x in enumerate(col) if x],
                "squared_norm": _rational_text(sum((x * x for x in col), Fraction(0))),
                "zero": not any(col),
            }
            for (name, _), col in zip(outputs, columns)
        ],
        "image_rank": len(data["basis_columns"]),
        "decoder": data,
    }


def replay(spec, certificate, limits=None):
    """Recompute exactly and require matching coordinates, decoder and provenance.

    The same explicit resource limits as :func:`evaluate` apply. Refusal is
    inconclusive; only a complete matching recomputation verifies the artifact.
    """
    actual = evaluate(spec, limits)
    require(actual == certificate, "certificate mismatch on exact recomputation")
    return actual


def compile_graph_block_coordinates(
    composition,
    graph,
    *,
    channels,
    occupancy=None,
    max_vertices=5000,
    max_dimension=20000,
    max_outputs=128,
):
    """Compile every requested source column in one common PIT2 program.

    ``occupancy`` optionally restricts the source to a single fine cell, with
    one row per chronological block and one column per named channel. Separate
    calls have separate coordinate spaces; their columns must not be combined.
    Resource exhaustion raises :class:`ResourceLimit`, never a zero verdict.
    """
    from ..quotients import DependenceGraph

    if not isinstance(graph, DependenceGraph):
        raise TypeError("graph must be a DependenceGraph")
    spec = {
        "kind": "graph-block",
        "q": list(composition),
        "channels": channels,
        "edges": [list(edge) for edge in sorted(graph.edges)],
    }
    if occupancy is not None:
        spec["occupancy"] = [list(row) for row in occupancy]
    return evaluate(
        spec,
        {"vertices": max_vertices, "dimension": max_dimension, "outputs": max_outputs},
    )


def decode_coordinates(compilation, vector, *, max_dimension=20000, max_outputs=128):
    """Return exact image-basis coefficients, rejecting vectors outside the image.

    The result identifies a chosen source representative, not a unique original
    source vector when the source map has a kernel. Pass a compiler result or
    one returned by ``replay(spec, saved_result)``. Structural and minor checks
    here detect malformed data; they do not authenticate a saved source claim.
    Replay persisted/external results before decoding them.
    """
    require(isinstance(compilation, dict), "expected a compiled coordinate object")
    integer(max_dimension, "maximum decoder dimension", 1)
    integer(max_outputs, "maximum decoder outputs", 1)
    dimension = integer(compilation.get("dimension"), "coordinate dimension", 1)
    budget(dimension, max_dimension, "decoder coordinate dimension")
    require(
        compilation.get("schema") == 1
        and compilation.get("method") == "PIT2-tensor-first-column-v1",
        "unsupported coordinate schema",
    )
    raw_columns = compilation.get("columns")
    require(isinstance(raw_columns, list) and raw_columns, "invalid coordinate columns")
    budget(len(raw_columns), max_outputs, "decoder source outputs")
    vector = tuple(vector)
    require(len(vector) == dimension, "wrong coordinate dimension")
    values = [
        value if isinstance(value, Fraction) else rational(value) for value in vector
    ]
    columns = []
    for column in raw_columns:
        require(
            isinstance(column, dict) and isinstance(column.get("coordinates"), list),
            "invalid sparse coordinate column",
        )
        dense = [Fraction(0)] * dimension
        previous = -1
        for entry in column["coordinates"]:
            require(
                isinstance(entry, (list, tuple)) and len(entry) == 2,
                "invalid sparse coordinate entry",
            )
            index, value = entry
            integer(index, "coordinate index")
            require(
                previous < index < dimension,
                "coordinate indices must be unique, increasing and in range",
            )
            coefficient = rational(value)
            require(coefficient != 0, "sparse coordinates must be nonzero")
            dense[index] = coefficient
            previous = index
        columns.append(dense)
    rank = integer(compilation.get("image_rank"), "image rank")
    require(rank <= min(dimension, len(columns)), "invalid image rank")
    data = compilation.get("decoder")
    require(isinstance(data, dict), "invalid decoder matrix")
    for field, bound in (("basis_columns", len(columns)), ("pivot_rows", dimension)):
        indices = data.get(field)
        require(
            isinstance(indices, list) and len(indices) == rank,
            "invalid decoder index shape",
        )
        require(
            all(type(index) is int and 0 <= index < bound for index in indices),
            "decoder indices must be in range",
        )
        require(len(set(indices)) == rank, "decoder indices must be unique")
    inverse = data.get("inverse")
    require(
        isinstance(inverse, list) and len(inverse) == rank,
        "invalid decoder inverse shape",
    )
    require(
        all(isinstance(row, list) and len(row) == rank for row in inverse),
        "invalid decoder inverse shape",
    )
    parsed = [[rational(x) for x in row] for row in inverse]
    # The selected minor must actually be invertible with the advertised inverse.
    # This also rejects a corrupted inverse when the requested vector is zero.
    minor = [[columns[j][i] for j in data["basis_columns"]] for i in data["pivot_rows"]]
    require(
        all(
            sum((a * minor[k][j] for k, a in enumerate(row)), Fraction()) == int(i == j)
            for i, row in enumerate(parsed)
            for j in range(rank)
        ),
        "decoder inverse does not invert the selected minor",
    )
    if not rank:
        require(
            not any(any(column) for column in columns),
            "zero-rank decoder has nonzero columns",
        )
    return tuple(decode_image(columns, data, values))


__all__ = [
    "ResourceLimit",
    "rational",
    "compile_graph_block_coordinates",
    "decode_coordinates",
    "evaluate",
    "replay",
]


def load_json(path):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite JSON number")

    return json.loads(
        path.read_text(),
        object_pairs_hook=object_pairs,
        parse_constant=nonfinite,
        parse_int=_decimal_integer,
    )
