# create_constraints_vector.py
import numpy as np

def Create_ConstraintsVector(Constraints, GlobalId):
    """
    Constructs U_P (NCons x 1) from Constraints (NCons x 3) and GlobalId.
    Constraint numbering MUST be consistent with how GlobalId was built.
    """
    Constraints = np.asarray(Constraints, dtype=float)
    if Constraints.ndim != 2 or Constraints.shape[1] != 3:
        raise ValueError("Constraints must be (NCons x 3) -> [node, dof, value]")

    # Determine number of constraints from GlobalId (max constraint id encountered)
    neg_vals = -GlobalId[GlobalId < 0]
    NCons = 0 if neg_vals.size == 0 else int(np.max(neg_vals))

    U_P = np.zeros((NCons,), dtype=float)

    # By construction in Create_ID_Matrix, constraint ids are assigned in the
    # given order of rows in Constraints: first -> id 1, second -> id 2, ...
    for i in range(Constraints.shape[0]):
        cval = float(Constraints[i, 2])
        U_P[i] = cval

    return U_P
