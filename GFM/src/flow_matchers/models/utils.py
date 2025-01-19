import torch


def binomial_coefficient(n, k):
    return torch.exp(
        torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    )


def bezier_point_and_derivative(x0, x1, intermediate_control_points, t):
    """
    Compute the point on the Bézier curve and its derivative w.r.t. t, separating x0 and x1 from intermediate control points.

    x0: Start points, Tensor of shape (batch_size, dim)
    x1: End points, Tensor of shape (batch_size, dim)
    intermediate_control_points: Tensor of shape (batch_size, num_intermediate_control_points, dim)
    t: Tensor of shape (batch_size, 1)

    Returns:
    - The point on the Bézier curve for each t (batch_size, dim)
    - The derivative of the Bézier curve at each t (batch_size, dim)
    """

    control_points = torch.cat(
        [x0.unsqueeze(1), intermediate_control_points, x1.unsqueeze(1)], dim=1
    )

    n = (
        control_points.shape[1] - 1
    )  # Degree of the Bézier curve, adjusted for x0 and x1
    t = t.squeeze(-1)
    t_expanded = t.unsqueeze(-1).expand(-1, control_points.shape[1])
    powers_t = torch.pow(
        t_expanded, torch.arange(n + 1, device=control_points.device).float()
    )
    powers_one_minus_t = torch.pow(
        1 - t_expanded, torch.arange(n, -1, -1, device=control_points.device).float()
    )

    # Compute binomial coefficients
    binom_coeffs = torch.tensor(
        [
            binomial_coefficient(torch.tensor([n]), torch.tensor([i]))
            for i in range(n + 1)
        ],
        device=control_points.device,
    ).float()

    # Bernstein basis
    bernstein_basis = binom_coeffs * powers_t * powers_one_minus_t

    # Point on the Bézier curve
    bezier_points = torch.sum(bernstein_basis.unsqueeze(-1) * control_points, dim=1)

    # Derivative of the Bézier curve
    # Compute the binomial coefficients for n-1
    binom_coeffs_diff = torch.tensor(
        [
            binomial_coefficient(torch.tensor([n - 1]), torch.tensor([i]))
            for i in range(n)
        ],
        device=control_points.device,
    ).float()  # Shape: (1, n)

    # Compute powers of t and (1 - t)
    t_powers_diff = powers_t[:, :-1]
    one_minus_t_powers_diff = powers_one_minus_t[:, 1:]

    # Compute Bernstein basis polynomials of degree n-1
    bernstein_basis_diff = (
        binom_coeffs_diff * t_powers_diff * one_minus_t_powers_diff
    )  # Shape: (batch_size, n)

    # Compute the differences between consecutive control points
    control_point_diffs = n * (
        control_points[:, 1:] - control_points[:, :-1]
    )  # Shape: (batch_size, n, dim

    # Calculate the Bézier curve derivative
    bezier_derivatives = torch.sum(
        bernstein_basis_diff.unsqueeze(-1) * control_point_diffs, dim=1
    )  # Shape: (batch_size, dim)

    return bezier_points, bezier_derivatives


def project_onto_line_torch(P0, P1, Pc):
    d = torch.sum((P0 - P1) ** 2)
    t = torch.sum((Pc - P0) * (P1 - P0)) / d
    return P0 + t * (P1 - P0)


def f_adjusted(P0, P1, Pc, alpha):
    proj = project_onto_line_torch(P0, P1, Pc)
    return alpha * (Pc - proj) + proj
