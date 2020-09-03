import torch


def apply_unit_sphere_normalization(x, dim=1):
    """
    Normalize an input x as x/norm(x, p=2).
    Normalizes over fp32 cast inputs.
    Replaces norm values equal or close to 0 with 1s.

    :param x: torch.Tensor
    :param dim: int, dim to normalize over
    """
    dtype = x.dtype
    x = x.to(torch.float32)
    norms = torch.norm(x, p=2, dim=dim, keepdim=True)
    # In the rare case that the model predicts all zeros
    # Ensure no div by zero here
    norms = torch.where(torch.isclose(norms, torch.zeros_like(norms)),
                        torch.ones_like(norms),
                        norms)
    x = x / norms
    return x.to(dtype)
