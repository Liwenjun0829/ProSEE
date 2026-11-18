"""Generate monomial bases whose exponent vectors sum to zero.

The source pool is always generated in four variables with exponent sum zero.
Lower-arity pools are obtained by setting trailing variables to one and then
discarding the all-zero projection and deduplicating it. Public functions are
side-effect free so fitting pipelines can import this module safely.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable, Sequence

import sympy as sp


def generate_exponent_vectors(
    num_variables: int, max_abs: int = 1
) -> list[tuple[int, ...]]:
    """Return all non-zero, zero-sum integer exponent vectors."""
    if num_variables < 1:
        raise ValueError("num_variables must be at least 1")
    if max_abs < 1:
        raise ValueError("max_abs must be at least 1")
    vectors = (
        alpha
        for alpha in itertools.product(
            range(-max_abs, max_abs + 1), repeat=num_variables
        )
        if any(alpha) and sum(alpha) == 0
    )
    return sorted(vectors)


def project_four_variable_exponents(
    num_variables: int,
    max_abs: int = 1,
) -> list[tuple[int, ...]]:
    """Project the 4-D pool by setting trailing variables to one and deduplicate.

    For example, source exponent ``(a, b, c, d)`` becomes ``(a, b)`` for
    two-variable data. Projections whose retained exponents are all zero are
    removed.
    """
    if not 1 <= num_variables <= 4:
        raise ValueError("num_variables must be between 1 and 4")
    source = generate_exponent_vectors(4, max_abs)
    projected = {
        alpha[:num_variables]
        for alpha in source
        if any(alpha[:num_variables])
    }
    return sorted(projected)


def exponent_vector_to_expr(
    alpha: Sequence[int], variable_names: Sequence[str] | None = None
) -> sp.Expr:
    """Convert one exponent vector into a SymPy monomial."""
    names = list(variable_names or (f"x{i + 1}" for i in range(len(alpha))))
    if len(names) != len(alpha):
        raise ValueError("variable_names and alpha must have the same length")
    symbols = sp.symbols(" ".join(names), seq=True)
    return sp.Mul(*(
        symbol ** int(power) for symbol, power in zip(symbols, alpha)
    ))


def generate_zero_sum_bases(
    num_variables: int,
    max_abs: int = 1,
    variable_names: Sequence[str] | None = None,
) -> list[sp.Expr]:
    """Generate one arity by projecting and deduplicating the 4-D pool."""
    return [
        exponent_vector_to_expr(alpha, variable_names)
        for alpha in project_four_variable_exponents(
            num_variables, max_abs
        )
    ]


def save_to_file(
    filename: str | Path,
    solutions: Iterable[Sequence[int]],
    variable_names: Sequence[str],
) -> None:
    """Save exponent vectors as readable SymPy-style expressions."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    expressions = [
        str(exponent_vector_to_expr(alpha, variable_names))
        for alpha in solutions
    ]
    path.write_text(
        "\n".join(expressions) + ("\n" if expressions else ""),
        encoding="utf-8",
    )


def generate_and_save_all(
    output_directory: str | Path = ".",
    max_variables: int = 4,
    max_abs: int = 1,
    variable_start_index: int = 0,
) -> dict[int, list[sp.Expr]]:
    """Generate all requested arities with legacy filenames/x0 indexing."""
    if not 1 <= max_variables <= 4:
        raise ValueError("max_variables must be between 1 and 4")
    output = Path(output_directory)
    generated: dict[int, list[sp.Expr]] = {}
    for arity in range(1, max_variables + 1):
        names = [f"x{i + variable_start_index}" for i in range(arity)]
        vectors = project_four_variable_exponents(
            arity, max_abs
        )
        expressions = [exponent_vector_to_expr(alpha, names) for alpha in vectors]
        generated[arity] = expressions
        suffix = "origin" if arity == 4 else f"{arity}vari"
        save_to_file(
            output / f"zero_sum_bases_{suffix}.txt", vectors, names
        )
        print(f"Found {len(expressions)} items {arity}-variable zero-sum bases")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate zero-sum monomial bases of different arities")
    parser.add_argument("--output-directory", default=".")
    parser.add_argument("--max-variables", type=int, default=4)
    parser.add_argument("--max-abs", type=int, default=1)
    parser.add_argument(
        "--variable-start-index", type=int, default=0,
        help="Starting variable index in text output; default 0 is compatible with the original script",
    )
    args = parser.parse_args()
    generate_and_save_all(
        args.output_directory,
        args.max_variables,
        args.max_abs,
        args.variable_start_index,
        not args.disable_binary_reciprocals,
    )


if __name__ == "__main__":
    main()
