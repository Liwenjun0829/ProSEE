import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import sympy as sp

CONST = sp.Symbol("const")


def parse_formula(text: str) -> sp.Expr:
    return sp.sympify(text, locals={"const": CONST})


def replace_constants(expr: sp.Expr) -> sp.Expr:
    """Replace numeric leaves, including signed numeric coefficients, by const."""
    return expr.xreplace({n: CONST for n in sp.preorder_traversal(expr) if n.is_Number})


def canonical_structure(text_or_expr) -> str:
    expr = parse_formula(text_or_expr) if isinstance(text_or_expr, str) else text_or_expr
    expr = replace_constants(expr)
    # srepr retains the tree and SymPy's canonical ordering; variable numbers remain useful.
    return sp.srepr(expr)


def node_token(node: sp.Expr) -> str:
    if node == CONST or node.is_Number:
        return "CONST"
    if node.is_Symbol:
        return "VAR_" + re.sub(r"\d+", "N", node.name)
    return f"{node.func.__name__}/{len(node.args)}"


@dataclass
class TreeEncoder:
    max_nodes: int = 96
    vocab: Optional[Dict[str, int]] = None

    def fit(self, formulas: Iterable[str]):
        tokens = {node_token(n) for f in formulas for n in sp.preorder_traversal(replace_constants(parse_formula(f)))}
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.vocab.update({t: i + 2 for i, t in enumerate(sorted(tokens))})
        return self

    def encode(self, formula: str) -> List[int]:
        if self.vocab is None:
            raise RuntimeError("encoder is not fitted")
        ids = [self.vocab.get(node_token(n), 1) for n in sp.preorder_traversal(replace_constants(parse_formula(formula)))]
        return (ids[: self.max_nodes] + [0] * self.max_nodes)[: self.max_nodes]

    def to_dict(self):
        return {"max_nodes": self.max_nodes, "vocab": self.vocab, "encoding": "sympy_preorder_padded"}

    @classmethod
    def from_dict(cls, value):
        return cls(int(value["max_nodes"]), dict(value["vocab"]))
