"""Compute n! recursively."""


def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("factorial undefined for negatives")
    return n * factorial(n)
