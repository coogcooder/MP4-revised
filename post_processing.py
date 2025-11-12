# post_processing.py
import numpy as np

def PostProcessing(GlobalId, U_F, U_P):
    """
    Assemble full solution U (NumNodes x dofs_per_node) from:
      - GlobalId: +eqn id for free dof, -constraint id for prescribed dof
      - U_F: (NEqns,) values for free dofs
      - U_P: (NCons,) values for prescribed dofs
    """
    GlobalId = np.asarray(GlobalId, dtype=int)
    U = np.zeros_like(GlobalId, dtype=float)

    for i in range(GlobalId.shape[0]):
        for j in range(GlobalId.shape[1]):
            gid = GlobalId[i, j]
            if gid > 0:
                U[i, j] = U_F[gid - 1]
            elif gid < 0:
                U[i, j] = U_P[(-gid) - 1]
            else:
                raise ValueError("GlobalId contains a zero entry (unassigned DOF).")

    return U
