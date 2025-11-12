# calculate_global_matrices.py
import numpy as np
from gauss_quadrature import GaussPoints
from calculate_local_matrices import CalculateLocalMatrices
from assemble import Assemble

def CalculateGlobalMatrices(Connectivity, Coord, diffusivity_function,
                            dim, dofs_per_node, EleType, GlobalId,
                            load_type, NCons, Nele, NEqns, NGPTS):
    """
    Builds (K_FF, K_FP, R_F) by looping elements.
    """
    Connectivity = np.asarray(Connectivity, dtype=int)
    Coord = np.asarray(Coord, dtype=float)

    # Allocate globals
    K_FF = np.zeros((NEqns, NEqns), dtype=float)
    K_FP = np.zeros((NEqns, NCons), dtype=float)
    R_F  = np.zeros((NEqns,), dtype=float)

    # Gauss points (in parent space)
    r, w = GaussPoints(dim, EleType, NGPTS)  # r: (nGP x dim), w: (nGP,)

    # Element loop
    for e in range(Nele):
        EleNodes = Connectivity[e, :]
        # Build xCap for this element: MATLAB-style 1-based EleNodes -> Python 0-based
        xCap = Coord[EleNodes - 1, :]  # (neln x dim)

        Klocal, rlocal = CalculateLocalMatrices(diffusivity_function,
                                                dofs_per_node, EleNodes, EleType,
                                                load_type, r, w, xCap)

        K_FF, K_FP, R_F = Assemble(dofs_per_node, EleNodes, GlobalId,
                                   Klocal, rlocal, K_FF, K_FP, R_F)

    return K_FF, K_FP, R_F
