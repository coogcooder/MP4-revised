"""Driver routine for the steady diffusion finite-element solver.

This module mirrors the provided MATLAB function `Driver_Steady_Diffusion.m`
and glues together the helper utilities available in this repository.  The
function validates the user-supplied mesh/constraint data, assembles the global
system, solves for the free degrees of freedom and finally reconstructs the
full nodal field (free + prescribed values).
"""

from __future__ import annotations

import numpy as np

from calculate_global_matrices import CalculateGlobalMatrices
from create_constraints_vector import Create_ConstraintsVector
from create_ID_matrix import Create_ID_Matrix
from post_processing import PostProcessing


def Driver_Steady_Diffusion(
    Connectivity,
    Constraints,
    Coord,
    diffusion_function,
    dim,
    dofs_per_node,
    EleType,
    load_type,
    NCons,
    Nele,
    NGPTS,
    NumNodes,
):
    """Solve a steady diffusion problem for the supplied finite-element mesh.

    Parameters
    ----------
    Connectivity : array_like, shape (Nele, neln)
        Element connectivity expressed with 1-based node numbers.
    Constraints : array_like, shape (NCons, 3)
        Dirichlet constraints ``[node_id, dof_id, value]``.  The dof numbering
        follows the MATLAB convention (starting at one).  Pass an empty array if
        no constraints are prescribed.
    Coord : array_like, shape (NumNodes, dim)
        Global coordinates of all mesh nodes.
    diffusion_function : dict
        Description of the diffusivity field, see :func:`get_dmat.Get_DMat`.
    dim : int
        Spatial dimension of the problem (1, 2, ...).
    dofs_per_node : int
        Number of degrees of freedom per node.
    EleType : str
        Element type identifier recognised by :func:`shape_functions.ShapeFunctions`.
    load_type : dict or None
        Volumetric load description, see :func:`get_volumetric_source.Get_VolumetricSource`.
    NCons : int
        Number of prescribed (Dirichlet) constraints expected in ``Constraints``.
    Nele : int
        Total number of elements in the mesh.
    NGPTS : int
        Number of Gauss points per direction used for numerical integration.
    NumNodes : int
        Total number of nodes in the mesh.

    Returns
    -------
    numpy.ndarray, shape (NumNodes, dofs_per_node)
        Nodal solution including both free and prescribed values.

    Raises
    ------
    ValueError
        If any of the supplied arguments are inconsistent or invalid.
    """

    # ------------------------------------------------------------------
    # Basic validation of scalar inputs
    # ------------------------------------------------------------------
    for name, value in {
        "dim": dim,
        "dofs_per_node": dofs_per_node,
        "NCons": NCons,
        "Nele": Nele,
        "NGPTS": NGPTS,
        "NumNodes": NumNodes,
    }.items():
        if not isinstance(value, (int, np.integer)):
            raise ValueError(f"ERROR: {name} must be an integer (received {type(value)!r}).")
        if name in {"dim", "dofs_per_node", "Nele", "NGPTS", "NumNodes"} and value <= 0:
            raise ValueError(f"ERROR: {name} must be a positive integer (received {value}).")
        if name == "NCons" and value < 0:
            raise ValueError("ERROR: NCons must be zero or a positive integer.")

    if not isinstance(EleType, str) or not EleType:
        raise ValueError("ERROR: EleType must be a non-empty string identifier.")

    # ------------------------------------------------------------------
    # Convert array-like inputs and validate their dimensions
    # ------------------------------------------------------------------
    Connectivity = np.asarray(Connectivity, dtype=int)
    if Connectivity.ndim != 2:
        raise ValueError("ERROR: Connectivity must be a 2-D array with shape (Nele, neln).")
    if Connectivity.shape[0] != Nele:
        raise ValueError(
            "ERROR: Nele does not match the number of rows in Connectivity: "
            f"{Nele} vs {Connectivity.shape[0]}."
        )

    Coord = np.asarray(Coord, dtype=float)
    if Coord.ndim != 2:
        raise ValueError("ERROR: Coord must be a 2-D array with shape (NumNodes, dim).")
    if Coord.shape[0] != NumNodes:
        raise ValueError(
            "ERROR: NumNodes does not match the number of nodes in Coord: "
            f"{NumNodes} vs {Coord.shape[0]}."
        )
    if Coord.shape[1] != dim:
        raise ValueError(
            "ERROR: dim does not match the coordinate dimension: "
            f"{dim} vs {Coord.shape[1]}."
        )

    Constraints = np.asarray(Constraints, dtype=float)
    if Constraints.size == 0:
        Constraints = Constraints.reshape(0, 3)
    if Constraints.ndim != 2 or Constraints.shape[1] != 3:
        raise ValueError("ERROR: Constraints must be an (NCons x 3) array.")
    if Constraints.shape[0] != NCons:
        raise ValueError(
            "ERROR: NCons does not match the number of constraint rows: "
            f"{NCons} vs {Constraints.shape[0]}."
        )

    if diffusion_function is None:
        raise ValueError("ERROR: diffusion_function cannot be None.")

    if load_type is not None and not isinstance(load_type, dict):
        raise ValueError("ERROR: load_type must be a dictionary (or None).")

    # ------------------------------------------------------------------
    # Build global ID mapping and assemble the system of equations
    # ------------------------------------------------------------------
    GlobalId, NEqns = Create_ID_Matrix(Constraints, dofs_per_node, NCons, NumNodes)

    K_FF, K_FP, R_F = CalculateGlobalMatrices(
        Connectivity,
        Coord,
        diffusion_function,
        dim,
        dofs_per_node,
        EleType,
        GlobalId,
        load_type,
        NCons,
        Nele,
        NEqns,
        NGPTS,
    )

    # ------------------------------------------------------------------
    # Build the constraint vector, solve for free DOFs, and post-process
    # ------------------------------------------------------------------
    U_P = Create_ConstraintsVector(Constraints, GlobalId)

    if NEqns > 0:
        rhs = R_F - (K_FP @ U_P)
        try:
            U_F = np.linalg.solve(K_FF, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "ERROR: Failed to solve the linear system for free DOFs."
            ) from exc
    else:
        # No free DOFs: ensure the system is self-consistent
        rhs = R_F - (K_FP @ U_P)
        if not np.allclose(rhs, 0.0):
            raise ValueError(
                "ERROR: Inconsistent system – no free DOFs but the reduced right-hand side is non-zero."
            )
        U_F = np.zeros((0,), dtype=float)

    U = PostProcessing(GlobalId, U_F, U_P)
    return U


__all__ = ["Driver_Steady_Diffusion"]