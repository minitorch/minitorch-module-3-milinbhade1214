from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
from collections import defaultdict

# ## Task 1.1
# Central Difference calculation

from . import operators

def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals = list(vals)
    # print(vals)
    vals[arg] += epsilon/2
    f1 = f(*vals)
    vals[arg] -= epsilon
    # print(vals)
    f2 = f(*vals)
    # print("Values: ", f1, f2, epsilon)
    return (f1 - f2)/epsilon 



variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass

def is_constant(val):
    return val.is_constant() or val.history is None


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    sorted = []
    visited = set()

    def visit(var):
        if var.unique_id in visited:
            return
        if not var.is_leaf():
            for input in var.history.inputs:
                if not is_constant(input):
                    visit(input)
        visited.add(var.unique_id)
        sorted.insert(0, var)

    visit(variable)
    return sorted
    
            
    


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted = topological_sort(variable)

    d_dict = defaultdict(float)
    d_dict[variable.unique_id] = deriv
    for var in sorted:
        d = d_dict[var.unique_id]
        if not var.is_leaf():
            for v, d_part in var.chain_rule(d):
                d_dict[v.unique_id] += d_part
        else:
            var.accumulate_derivative(d)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
