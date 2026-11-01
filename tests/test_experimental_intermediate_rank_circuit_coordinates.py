from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

import tensorsig.experimental.offset_coded_signatures.circuit_coordinates as circuit_coordinates
from tensorsig.experimental.offset_coded_signatures.circuit_coordinates import (
    CircuitCoordinatePlan,
    compile_tree_circuit_coordinates,
    signed_unicyclic_relations,
)
from tensorsig.offset_coded_signatures._graph.tree_moments import (
    tree_moment,
)


def _compositions(total: int) -> tuple[tuple[int, ...], ...]:
    if total == 0:
        return ((),)
    return tuple(
        (first, *rest)
        for first in range(1, total + 1)
        for rest in _compositions(total - first)
    )


def test_triangle_relation_has_global_arrow_signs() -> None:
    (relation,) = signed_unicyclic_relations(3)
    triangle = frozenset({(0, 1), (0, 2), (1, 2)})

    assert {
        next(iter(triangle - tree)): coefficient for tree, coefficient in relation
    } == {
        (0, 1): 1,
        (1, 2): 1,
        (0, 2): -1,
    }


def test_relation_iterator_yields_before_exhausting_cayley_trees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_tree = frozenset({(0, 1), (0, 2)})

    def guarded_trees(degree: int):
        assert degree == 3
        yield first_tree
        raise AssertionError("Cayley trees were eagerly materialized")

    monkeypatch.setattr(circuit_coordinates, "_labelled_trees", guarded_trees)

    assert next(signed_unicyclic_relations(3))


def test_unicyclic_relation_rows_have_tree_support() -> None:
    relations = tuple(signed_unicyclic_relations(4))

    assert len(relations) == 15
    for relation in relations:
        trees = tuple(tree for tree, _ in relation)
        unicyclic = frozenset().union(*trees)
        assert len(relation) >= 3
        assert len(set(trees)) == len(trees)
        assert len(unicyclic) == 4
        assert all(len(tree) == 3 and len(unicyclic - tree) == 1 for tree in trees)
        assert all(coefficient in {-1, 1} for _, coefficient in relation)


@pytest.mark.parametrize(
    ("degree", "tree_count", "relation_count", "rank", "basis_count", "nonzeros"),
    [
        (2, 1, 0, 0, 1, 0),
        (3, 3, 1, 1, 2, 3),
        (4, 16, 15, 10, 6, 48),
        (5, 125, 222, 101, 24, 750),
        (6, 1296, 3660, 1176, 120, 12960),
    ],
)
def test_exact_circuit_coordinate_censuses(
    degree: int,
    tree_count: int,
    relation_count: int,
    rank: int,
    basis_count: int,
    nonzeros: int,
) -> None:
    plan = compile_tree_circuit_coordinates(degree)

    assert len(plan.trees) == tree_count
    assert plan.raw_relation_count == relation_count
    assert plan.relation_rank == rank
    assert len(plan.basis_trees) == basis_count
    assert plan.raw_relation_nonzeros == nonzeros
    assert plan.max_row_width <= tree_count
    assert plan.max_numerator_bits >= int(bool(relation_count))
    assert plan.max_denominator_bits >= int(bool(relation_count))


def test_basis_values_reconstruct_direct_tree_moments_through_degree_four() -> None:
    for degree in range(2, 5):
        plan = compile_tree_circuit_coordinates(degree)
        for composition in _compositions(degree):
            direct = tuple(tree_moment(composition, tree) for tree in plan.trees)
            basis_values = tuple(direct[index] for index in plan.basis_indices)
            assert plan.reconstruct(basis_values) == direct


def test_plan_is_frozen_and_validates_inputs() -> None:
    plan = compile_tree_circuit_coordinates(3)

    with pytest.raises(FrozenInstanceError):
        plan.degree = 4  # type: ignore[misc]
    with pytest.raises(ValueError, match="expected 2 basis values"):
        plan.reconstruct((Fraction(1),))

    for degree in (1, 0, -1):
        with pytest.raises(ValueError, match="at least two"):
            compile_tree_circuit_coordinates(degree)
        with pytest.raises(ValueError, match="at least two"):
            tuple(signed_unicyclic_relations(degree))
    for degree in (True, 2.0):
        with pytest.raises(TypeError, match="integer"):
            compile_tree_circuit_coordinates(degree)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="integer"):
            tuple(signed_unicyclic_relations(degree))  # type: ignore[arg-type]


def test_plan_metrics_are_frozen_result_data() -> None:
    plan = compile_tree_circuit_coordinates(3)

    assert isinstance(plan, CircuitCoordinatePlan)
    assert plan.reconstruct((1, 2)) == (
        Fraction(-1),
        Fraction(1),
        Fraction(2),
    )
