# assemble.py
import numpy as np

def Assemble(dofs_per_node, EleNodes, GlobalId, Klocal, rlocal, K_FF, K_FP, R_F):
    """
    Accumulates an element's local matrices into global (FF/FP) and R_F.

    GlobalId: (NumNodes x dofs_per_node), positive=free eqn id (1..NEqns), negative= -constraint id (1..NCons)
    EleNodes: indices of the element's nodes (1-based as in MATLAB). Converted to 0-based here.
    """
    EleNodes = np.asarray(EleNodes, dtype=int).ravel()
    neln = EleNodes.size

    # Map local dofs -> (node_idx, dof_idx)
    for a in range(neln):
        node_a = EleNodes[a] - 1  # to 0-based
        for p in range(dofs_per_node):
            la = a * dofs_per_node + p
            gid_a = GlobalId[node_a, p]

            # Load vector assembly only into free rows
            if gid_a > 0:
                R_F[gid_a - 1] += rlocal[la]

            # Stiffness entries
            for b in range(neln):
                node_b = EleNodes[b] - 1
                for q in range(dofs_per_node):
                    lb = b * dofs_per_node + q
                    gid_b = GlobalId[node_b, q]

                    if gid_a > 0 and gid_b > 0:
                        # (free, free)
                        K_FF[gid_a - 1, gid_b - 1] += Klocal[la, lb]
                    elif gid_a > 0 and gid_b < 0:
                        # (free, pres)
                        col = (-gid_b) - 1
                        K_FP[gid_a - 1, col] += Klocal[la, lb]
                    # If gid_a < 0 (prescribed row), it is not part of FF system; skip here.

    return K_FF, K_FP, R_F
