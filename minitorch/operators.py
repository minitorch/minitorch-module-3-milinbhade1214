"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: The product of x and y.
    """
    return x * y


def id(x: float) -> float:
    """Identity function.

    Args:
        x (float): A number.

    Returns:
        float: The same number.
    """
    return x


def add(x: float, y: float) -> float:
    ":math:`f(x, y) = x + y`"
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
        x (float): A number.

    Returns:
        float: The negation of x.
    """
    return -float(x)


def lt(x: float, y: float) -> float:
    """Less than comparison.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: 1.0 if x is less than y, else 0.0.
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Equality comparison.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: 1.0 if x is equal to y, else 0.0.
    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Maximum of two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: The greater number between x and y.
    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if two numbers are close to each other.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: 1.0 if |x - y| < 1e-2, else 0.0.
    """
    return 1.0 if abs(x - y) < 1e-2 else 0.0


EPS = 1e-6


def sigmoid(x: float) -> float:
    """Sigmoid function.

    Args:
        x (float): A number.

    Returns:
        float: The sigmoid of x.
    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))

def exp(x: float) -> float:
    """Exponential function.

    Args:
        x (float): A number.

    Returns:
        float: The exponential of x.
    """
    return math.exp(x)


def relu(x: float) -> float:
    """Rectified Linear Unit function.

    Args:
        x (float): A number.

    Returns:
        float: x if x is greater than 0, else 0.
    """
    return float(x) if x > 0 else 0.0
def log(x: float) -> float:
    """Logarithm of x
    Args:
        x (float): a number
    Returns:
        float: log(x)
    """
    return math.log(x)


def log_back(x: float, d: float) -> float:
    """Backward pass for logarithm function.

    Args:
        x (float): Input to the log function.
        d (float): The derivative of the log function with respect to its input.

    Returns:
        float: The gradient of the log function.
    """
    return d / (x + EPS)


def inv(x: float) -> float:
    """Inverse function.

    Args:
        x (float): A number.

    Returns:
        float: The inverse of x.
    """
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Backward pass for inverse function.

    Args:
        x (float): Input to the inverse function.
        d (float): The derivative of the inverse function with respect to its input.

    Returns:
        float: The gradient of the inverse function.
    """
    return -d / (x**2)


def relu_back(x: float, d: float) -> float:
    """
    Backward pass for ReLU function.
    Args:
        x (float): Input to the ReLU function.
        d (float): The derivative of the ReLU function with respect to its input.
    Returns:
        float: The gradient of the ReLU function.
    """
    return d if x > 0 else 0

def sigmoid_back(x: float, d: float) -> float:
    """
    Backward pass for Sigmoid
    """
    return sigmoid(x) * (1 - sigmoid(x)) * d

def relu_back(x: float, d:float) -> float:
    r"If :math:`f = relu` compute d :math:`d \times f'(x)`"
    return d if x > 0 else 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Map a function `fn` over an iterable.
    Args:
        fn: A function that takes a float and returns a float.
    Returns:
        A function that takes an iterable of floats and returns an iterable of floats.
    """

    def apply(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]

    return apply


def negList(ls: Iterable[float]) -> Iterable[float]:
    """
    Negate all elements of a list.
    Args:
        ls: A list of numbers.
    Returns:
        A list of negated numbers.
    """
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Map a function `fn` over two iterables.
    Args:
        fn: A function that takes two floats and returns a float.
    Returns:
        A function that takes two iterables of floats and returns an iterable of floats.
    """

    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return apply


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Add two lists element-wise.
    Args:
        ls1: A list of numbers.
        ls2: A list of numbers.
    Returns:
        A list of numbers that are the sum of the corresponding elements of ls1 and ls2.
    """
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """
    Higher order function to reduce a list.
    Args:
        fn: A function that takes two floats and returns a float.
        start: The initial value.
    Returns:
        A function that takes an iterable of floats and returns a float.
    """

    def apply(ls: Iterable[float]) -> float:
        acc = start
        for x in ls:
            acc = fn(acc, x)
        return acc

    return apply


def sum(ls: Iterable[float]) -> float:
    """
    Returns the sum of a list using `reduce` and `add`.
    Args:
        ls: A list of numbers.
    Returns:
        The sum of the list.
    """
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    """
    Product of a list of numbers.
    Args:
        ls: A list of numbers.
    Returns:
        product of the list.
    """
    return reduce(mul, 1)(ls)
