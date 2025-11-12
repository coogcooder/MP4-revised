# calculate_local_matrices.py
import numpy as np
from shape_functions import ShapeFunctions
from get_dmat import Get_DMat
from get_volumetric_source import Get_VolumetricSource

def CalculateLocalMatrices(diffusivity_function,
                           dofs_per_node,
                           EleNodes,
                           EleType,
                           load_type,
                           r, w, xCap):
    """
    Computes local stiffness (Klocal) and local load (rlocal) for one element.

    Assumes scalar diffusion (dofs_per_node == 1).
    xCap: (nNode_e x dim) nodal coordinates for THIS element (in global coords).
    r:   (nGauss x dim) Gauss points in parent space (zeta)
    w:   (nGauss,) Gauss weights
    """
    if dofs_per_node != 1:
        raise NotImplementedError("This implementation currently supports scalar problems (dofs_per_node == 1).")

    xCap = np.asarray(xCap, dtype=float)
    dim = xCap.shape[1]
    neln = xCap.shape[0]
    nldofs = neln * dofs_per_node

    Klocal = np.zeros((nldofs, nldofs), dtype=float)
    rlocal = np.zeros((nldofs,), dtype=float)

    # Loop Gauss points
    for igp, zeta in enumerate(np.atleast_2d(r)):
        # Shape functions & parent-space gradients
        N, dN_dz = ShapeFunctions(EleType, zeta)  # Expect: N (1 x neln) or (neln,), dN_dz (dim x neln)
        N = np.asarray(N, dtype=float).reshape(-1)        # (neln,)
        dN_dz = np.asarray(dN_dz, dtype=float)            # (dim x neln)

# ==========================================================
# Compute Jacobian and spatial derivatives
# ==========================================================
        dN_dz = np.asarray(dN_dz, dtype=float)
        neln = xCap.shape[0]
        dim = xCap.shape[1]

# Check if derivatives are shaped (neln x dim) or (dim x neln)
        if dN_dz.shape == (dim, neln):
    # Case 1: (dim x neln)
            J = dN_dz @ xCap  # (dim x neln) @ (neln x dim) = (dim x dim)
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError("Non-positive Jacobian determinant encountered.")
            invJ = np.linalg.inv(J)
    # Spatial derivatives (neln x dim)
            dN_dx = (invJ.T @ dN_dz).T

        elif dN_dz.shape == (neln, dim):
    # Case 2: (neln x dim)
            J = xCap.T @ dN_dz  # (dim x neln) @ (neln x dim) = (dim x dim)
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError("Non-positive Jacobian determinant encountered.")
            invJ = np.linalg.inv(J)
    # Spatial derivatives (neln x dim)
            dN_dx = dN_dz @ invJ

        else:
            raise ValueError(
                f"Unexpected dN_dz shape {dN_dz.shape}. Expected ({dim},{neln}) or ({neln},{dim})."
)


        # D at current physical point x = sum_i N_i * x_i
        x_phys = N @ xCap  # (dim,)

        D = Get_DMat(diffusivity_function, x_phys, dim=dim)  # (dim x dim)

        # Klocal contribution: ∫ (gradN)^T D (gradN) dV
        #   gradN: (dim x neln) -> we'll use dN_dx.T (dim x neln)
        # Build B = gradN (dim x neln)
        B = dN_dx.T  # (dim x neln)
        k_e = (B.T @ D @ B) * detJ * w[igp]  # (neln x neln)
        Klocal += k_e

        # rlocal contribution: ∫ N^T f dV
        fval = Get_VolumetricSource(load_type, x_phys)  # scalar
        rlocal += (N * fval) * detJ * w[igp]

    return Klocal, rlocal
