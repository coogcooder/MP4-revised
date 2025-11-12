import numpy as np
from create_ID_matrix import Create_ID_Matrix
from calculate_local_matrices import CalculateLocalMatrices
from calculate_global_matrices import CalculateGlobalMatrices
from create_constraints_vector import Create_ConstraintsVector
from post_processing import PostProcessing
from gauss_quadrature import GaussPoints
from driver_steady_diffusion import Driver_Steady_Diffusion


# ============================================================
# Utility to run a single test case
# ============================================================
def run_case(case_name, Connectivity, Coord, diffusivity_function, dim,
             dofs_per_node, EleType, Constraints, load_type, NGPTS):

    NumNodes = Coord.shape[0]
    NCons = Constraints.shape[0]
    Nele = Connectivity.shape[0]

    # Generate Global ID matrix
    GlobalId, NEqns = Create_ID_Matrix(Constraints, dofs_per_node, NCons, NumNodes)

    # Initialize global matrices
    K_FF, K_FP, R_F = CalculateGlobalMatrices(
        Connectivity, Coord, diffusivity_function, dim,
        dofs_per_node, EleType, GlobalId,
        load_type, NCons, Nele, NEqns, NGPTS
    )

    # =============================
    # Show local matrices for each element
    # =============================
    print("\n" + "="*80)
    print(f"CASE: {case_name}")
    print("="*80)

    r, w = GaussPoints(dim, EleType, NGPTS)
    for e in range(Nele):
        EleNodes = Connectivity[e, :]
        xCap = Coord[EleNodes - 1, :]
        Klocal, rlocal = CalculateLocalMatrices(
            diffusivity_function, dofs_per_node, EleNodes,
            EleType, load_type, r, w, xCap
        )
        print(f"\nElement {e+1} Local Stiffness Matrix (Klocal):\n{Klocal}")
        print(f"Element {e+1} Local Load Vector (rlocal):\n{rlocal}")

        # Build U_P vector and solve
    # =============================
    U_P = Create_ConstraintsVector(Constraints, GlobalId)
    rhs = R_F - (K_FP @ U_P)
    U_F = np.linalg.solve(K_FF, rhs) if NEqns > 0 else np.zeros((0,))
    U = PostProcessing(GlobalId, U_F, U_P)

    # =============================
    # Cross-check driver routine
    # =============================
    U_driver = Driver_Steady_Diffusion(
        Connectivity,
        Constraints,
        Coord,
        diffusivity_function,
        dim,
        dofs_per_node,
        EleType,
        load_type,
        NCons,
        Nele,
        NGPTS,
        NumNodes,
    )
    if not np.allclose(U, U_driver):
        raise AssertionError(
            "Driver_Steady_Diffusion results do not match the manual assembly solution."
        )
    
     # =============================
    # Global Results
    # =============================
    print("\nGlobalId (rows: nodes, cols: dofs):\n", GlobalId)
    print("\nGlobal Stiffness Matrix (K_FF):\n", K_FF)
    print("\nGlobal Load Vector (R_F):\n", R_F)
    print("\nPrescribed DOFs (U_P):\n", U_P)
    print("\nSolved Free DOFs (U_F):\n", U_F)

    # ============================================================
# FEM Test Driver: Q4 Element Diffusion Problem (3 cases)
# ============================================================
import numpy as np

from create_ID_matrix import Create_ID_Matrix
from calculate_local_matrices import CalculateLocalMatrices
from calculate_global_matrices import CalculateGlobalMatrices
from create_constraints_vector import Create_ConstraintsVector
from post_processing import PostProcessing
from gauss_quadrature import GaussPoints
from create_ID_matrix import Create_ID_Matrix
from calculate_local_matrices import CalculateLocalMatrices
from calculate_global_matrices import CalculateGlobalMatrices
from create_constraints_vector import Create_ConstraintsVector
from post_processing import PostProcessing
from gauss_quadrature import GaussPoints
from driver_steady_diffusion import Driver_Steady_Diffusion


# ============================================================
# Utility to run a single test case
# ============================================================
def run_case(case_name, Connectivity, Coord, diffusivity_function, dim,
             dofs_per_node, EleType, Constraints, load_type, NGPTS):

    NumNodes = Coord.shape[0]
    NCons = Constraints.shape[0]
    Nele = Connectivity.shape[0]

    # Generate Global ID matrix
    GlobalId, NEqns = Create_ID_Matrix(Constraints, dofs_per_node, NCons, NumNodes)

    # Initialize global matrices
    K_FF, K_FP, R_F = CalculateGlobalMatrices(
        Connectivity, Coord, diffusivity_function, dim,
        dofs_per_node, EleType, GlobalId,
        load_type, NCons, Nele, NEqns, NGPTS
    )

    # =============================
    # Show local matrices for each element
    # =============================
    print("\n" + "="*80)
    print(f"CASE: {case_name}")
    print("="*80)

    r, w = GaussPoints(dim, EleType, NGPTS)
    for e in range(Nele):
        EleNodes = Connectivity[e, :]
        xCap = Coord[EleNodes - 1, :]
        Klocal, rlocal = CalculateLocalMatrices(
            diffusivity_function, dofs_per_node, EleNodes,
            EleType, load_type, r, w, xCap
        )
        print(f"\nElement {e+1} Local Stiffness Matrix (Klocal):\n{Klocal}")
        print(f"Element {e+1} Local Load Vector (rlocal):\n{rlocal}")

    # =============================
    # Build U_P vector and solve
    # =============================
    U_P = Create_ConstraintsVector(Constraints, GlobalId)
    rhs = R_F - (K_FP @ U_P)
    U_F = np.linalg.solve(K_FF, rhs) if NEqns > 0 else np.zeros((0,))
    U = PostProcessing(GlobalId, U_F, U_P)
    # Build U_P vector and solve
    # =============================
    U_P = Create_ConstraintsVector(Constraints, GlobalId)
    rhs = R_F - (K_FP @ U_P)
    U_F = np.linalg.solve(K_FF, rhs) if NEqns > 0 else np.zeros((0,))
    U = PostProcessing(GlobalId, U_F, U_P)

    # =============================
    # Cross-check driver routine
    # =============================
    U_driver = Driver_Steady_Diffusion(
        Connectivity,
        Constraints,
        Coord,
        diffusivity_function,
        dim,
        dofs_per_node,
        EleType,
        load_type,
        NCons,
        Nele,
        NGPTS,
        NumNodes,
    )
    if not np.allclose(U, U_driver):
        raise AssertionError(
            "Driver_Steady_Diffusion results do not match the manual assembly solution."
        )

    # =============================
    # Global Results
    # =============================
    print("\nGlobalId (rows: nodes, cols: dofs):\n", GlobalId)
    print("\nGlobal Stiffness Matrix (K_FF):\n", K_FF)
    print("\nGlobal Load Vector (R_F):\n", R_F)
    print("\nPrescribed DOFs (U_P):\n", U_P)
    print("\nSolved Free DOFs (U_F):\n", U_F)
    print("\nFinal Assembled Solution Field (U):\n", U)
    print("="*80 + "\n")
    print("\nFinal Assembled Solution Field (U):\n", U)
    print("\nDriver Routine Solution Field (U_driver):\n", U_driver)
    print("="*80 + "\n")

    return {"case": case_name, "K_FF": K_FF, "R_F": R_F, "U": U}


# ============================================================
# Build and run test cases
# ============================================================
def build_cases():
    dim = 2
    EleType = 'Q4'
    dofs_per_node = 1
    NGPTS = 2

    # Common unit square mesh
    Coord = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=float)

    Connectivity = np.array([[1, 2, 3, 4]], dtype=int)

    cases = []