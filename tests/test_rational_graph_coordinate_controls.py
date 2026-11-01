"""Independent retained PIT2 word, normalization, decoder and replay controls."""

import json
from collections import defaultdict
from fractions import Fraction
from itertools import product
from math import prod

from tensorsig.offset_coded_signatures._graph.rational_coordinates import (
    ResourceLimit,
    coordinates,
    decode_image,
    dimensions,
    evaluate,
    graph_program,
    replay,
    require,
    validate_program,
)


def word_oracle(program, output):
    """Separate finite tensor-word expansion used only by bounded self-tests."""
    N, sizes, _, start, _, adjacency, order = program
    empty = ((),) * len(sizes)
    polynomials = [defaultdict(Fraction) for _ in range(N)]
    for u in reversed(order):
        polynomials[u][empty] += output[u]
        for v, letter, coefficient in adjacency[u]:
            for word, value in polynomials[v].items():
                target = list(word)
                if letter is not None:
                    j, i = letter
                    target[j] = (i,) + target[j]
                polynomials[u][tuple(target)] += coefficient * value
    answer = defaultdict(Fraction)
    for u in range(N):
        for word, value in polynomials[u].items():
            answer[word] += start[u] * value
    return {w: x for w, x in answer.items() if x}


def direct_word_coordinates(poly, dims):
    """Use the matrix-entry formula, independently of the fast recurrence."""
    answer = [Fraction(0)] * prod(dims)
    for words, coefficient in poly.items():
        factors = []
        for word, d in zip(words, dims):
            vector = [Fraction(1)] + [Fraction(0)] * (d - 1)
            for letter in reversed(word):
                vector = [
                    sum(
                        (
                            Fraction((-1) ** (u - v - 1), u * letter ** (u - v))
                            * vector[v]
                            for v in range(u)
                        ),
                        Fraction(0),
                    )
                    for u in range(d)
                ]
            factors.append(vector)
        for k, entries in enumerate(product(*factors)):
            answer[k] += coefficient * prod(entries)
    return answer


def self_test(limits):
    names = []

    def checked(name, condition):
        require(condition, f"control failed: {name}")
        names.append(name)

    def rejected(name, spec, bound=None):
        try:
            evaluate(spec, limits if bound is None else bound)
        except ValueError, KeyError, TypeError, ResourceLimit:
            names.append(name)
        else:
            raise ValueError(f"control did not reject: {name}")

    def abp(N, sizes, m, edges, endpoints):
        return {
            "kind": "tensor-abp",
            "vertices": N,
            "alphabet_sizes": sizes,
            "degree_bound": m,
            "edges": edges,
            "start": [1] + [0] * (N - 1),
            "outputs": [
                {"name": j, "weights": [int(k == j) for k in range(N)]}
                for j in endpoints
            ],
        }

    cases = [
        ("constant-and-N1", abp(1, [1], 0, [], [0])),
        (
            "scalar-cancellation",
            abp(2, [1], 0, [[0, 1, None, "1/2"], [0, 1, None, "-1/2"]], [1]),
        ),
        (
            "free-commutator",
            abp(
                4,
                [2],
                2,
                [
                    [0, 1, [0, 1], 1],
                    [1, 3, [0, 2], 1],
                    [0, 2, [0, 2], -1],
                    [2, 3, [0, 1], 1],
                ],
                [3],
            ),
        ),
        (
            "cross-factor-commutation",
            abp(
                4,
                [1, 1],
                1,
                [
                    [0, 1, [0, 1], 1],
                    [1, 3, [1, 1], 1],
                    [0, 2, [1, 1], -1],
                    [2, 3, [0, 1], 1],
                ],
                [3],
            ),
        ),
        (
            "tensor-product-nonzero",
            abp(3, [1, 1], 1, [[0, 1, [0, 1], 1], [1, 2, [1, 1], 1]], [2]),
        ),
        (
            "common-span-and-kernel",
            abp(
                4,
                [2],
                1,
                [
                    [0, 1, [0, 1], 1],
                    [0, 2, [0, 2], 1],
                    [1, 3, None, 1],
                    [2, 3, None, 1],
                ],
                [1, 2, 3],
            ),
        ),
    ]
    for name, spec in cases:
        p = validate_program(spec, limits)
        _, dims, _ = dimensions(p[0], p[1], p[2], limits["dimension"])
        columns = [coordinates(p, dims, out) for _, out in p[4]]
        for column, (_, out) in zip(columns, p[4]):
            checked(
                name + ":word-oracle",
                column == direct_word_coordinates(word_oracle(p, out), dims),
            )
        result = evaluate(spec, limits)
        checked(name + ":replay", replay(spec, result, limits) == result)
        if name == "free-commutator":
            checked("commutator-sign", columns[0][3] == Fraction(-1, 24))
            checked(
                "dimension-three-failure",
                direct_word_coordinates(word_oracle(p, p[4][0][1]), (3,)) == [0, 0, 0],
            )
        if name == "common-span-and-kernel":
            checked("common-span-rank", result["image_rank"] == 2)
            checked(
                "kernel-preserved",
                [a + b for a, b in zip(columns[0], columns[1])] == columns[2],
            )
            vector = [2 * a - 3 * b for a, b in zip(columns[0], columns[1])]
            checked(
                "decoder-combination",
                decode_image(columns, result["decoder"], vector) == [2, -3],
            )
            try:
                decode_image(
                    columns,
                    result["decoder"],
                    [Fraction(1)] + [Fraction(0)] * (len(vector) - 1),
                )
            except ValueError:
                names.append("decoder-outside-image")
            else:
                raise ValueError("outside-image vector accepted")
    for q in [[2], [1, 1], [1, 2]]:
        spec = {"kind": "graph-block", "q": q, "channels": 1, "edges": []}
        raw = graph_program(spec, limits)
        p = validate_program(raw, limits)
        poly = word_oracle(p, p[4][0][1])
        expected = (
            {}
            if q == [2]
            else (
                {((1, 2),): Fraction(1, 2), ((2, 1),): Fraction(-1, 2)}
                if q == [1, 1]
                else {
                    ((1, 2, 2),): Fraction(1, 6),
                    ((2, 1, 2),): Fraction(-1, 3),
                    ((2, 2, 1),): Fraction(1, 6),
                }
            )
        )
        checked(f"source-normalization-{q}", poly == expected)
        result = evaluate(spec, limits)
        checked(
            f"source-zero-predicate-{q}", result["columns"][0]["zero"] == (q == [2])
        )
        if q == [1, 1]:
            checked(
                "source-coordinate-sign",
                dict(result["columns"][0]["coordinates"])[3] == "-1/48",
            )
    empty = cases[0][1]
    result = evaluate(empty, limits)
    altered = dict(result, dimension=2)
    try:
        replay(empty, altered, limits)
    except ValueError:
        names.append("tampered-certificate")
    else:
        raise ValueError("tampered certificate accepted")
    rejected("bool-vertex-count", dict(empty, vertices=True))
    rejected("floating-coefficient", dict(empty, start=[0.5]))
    rejected("cycle", abp(1, [1], 0, [[0, 0, None, 1]], [0]))
    rejected("wrong-degree", abp(2, [1], 0, [[0, 1, [0, 1], 1]], [1]))
    rejected("bad-letter-index", abp(2, [1], 1, [[0, 1, [0, 0], 1]], [1]))
    rejected(
        "bad-occupancy",
        {"kind": "graph-block", "q": [2], "channels": 1, "occupancy": [[1]]},
    )
    rejected(
        "nonclique-cover",
        {
            "kind": "graph-block",
            "q": [1],
            "channels": 2,
            "edges": [],
            "cliques": [[0, 1]],
        },
    )
    rejected("dimension-resource-refusal", cases[2][1], dict(limits, dimension=1))
    print(
        json.dumps(
            {
                "artifact": "graph-block-rational-coordinate-evaluator",
                "controls": names,
                "count": len(names),
                "status": "passed",
            },
            sort_keys=True,
        )
    )


def test_retained_replay_controls(capsys):
    self_test({"vertices": 5000, "dimension": 20000, "outputs": 128})
    report = json.loads(capsys.readouterr().out)
    assert report["count"] == 36
    assert report["status"] == "passed"
