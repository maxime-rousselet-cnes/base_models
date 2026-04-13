"""
Base functions for symbolic models.
"""

from typing import Callable, Optional

from numpy import ndarray, zeros
from sympy import Expr, Matrix, MutableDenseMatrix, Symbol

from .runge_kutta_scheme import non_adaptive_runge_kutta_45


def evaluate_terminal_parameters(
    expression: Expr,
    parameter_expressions: dict[str, Expr],
    terminal_parameter_values: dict[str, float],
) -> Expr:
    """
    Substitudes terminal parameter expression into their values.
    """

    return expression.xreplace(
        rule={
            parameter_expressions[parameter_name]: value
            for parameter_name, value in terminal_parameter_values.items()
        }
    )


def partial_symbols(
    parameter: Expr, state_vector_line: list[Expr]
) -> tuple[list[Expr], MutableDenseMatrix]:
    """
    Generates the list of partial derivative symbols for a given parameter and state vector.
    """

    partials = [
        Symbol(
            r"\frac{\partial state_parameter}{\partial parameter}".replace(
                "state_parameter", str(state_parameter)
            ).replace("parameter", str(parameter))
        )
        for state_parameter in state_vector_line
    ]

    return partials, MutableDenseMatrix([[partial] for partial in partials])


def vector_variation_equation(
    dynamic: MutableDenseMatrix,
    parameter: Expr,
    partials: MutableDenseMatrix,
    state_vector_line: list[Expr],
) -> MutableDenseMatrix:
    """
    Applies the variation method to algebraically derive the time(-like)-dependent behavior of a
    partial derivative to integrate on the quadrature points.
    """

    return Matrix(
        [
            variation_equation(
                expression=expression,
                parameter=parameter,
                partials=partials,
                state_vector_line=state_vector_line,
            )
            for expression in dynamic.flat()
        ]
    )


def variation_equation(
    expression: Expr, parameter: Expr, partials: MutableDenseMatrix, state_vector_line: list[Expr]
) -> Expr:
    """
    Applies the variation method to algebraically derive a partial derivative expression with
    respect to a parameter.
    """

    return MutableDenseMatrix(
        [expression.diff(state_parameter) for state_parameter in state_vector_line]
    ).dot(b=partials) + expression.diff(parameter)


def fixed_timestep_integrator(
    fun: Callable,
    t: ndarray,
    y: ndarray,
    system_dimension: int = 6,
    i_parameter_initial_conditions: Optional[int] = None,
) -> ndarray:
    """
    Performs the numerical quadrature of partial derivatives over the integration's points.
    """

    dy_dgamma_0 = zeros(shape=system_dimension, dtype=complex)

    if (
        i_parameter_initial_conditions is not None
    ) and i_parameter_initial_conditions < system_dimension:

        # Because dx^i/dx^i_0(t=0) := 1.
        dy_dgamma_0[i_parameter_initial_conditions] = 1

    # Here, parameter represent what parameterizes a function call. In the case of the variation
    # equations, this is the already integrated state.
    return non_adaptive_runge_kutta_45(fun=fun, t=t, dy_dgamma_0=dy_dgamma_0, y=y)
