import numpy as np

def Create_ID_Matrix(Constraints, dofs_per_node, NCons, NumNodes):
    """
    CREATE_ID_MATRIX Maps global DOF numbers to equation or constraint numbers
    
    This function creates a mapping between global degrees of freedom (DOFs) and
    equation numbers for free DOFs or constraint numbers for constrained DOFs.
    Essential for finite element analysis assembly procedures.
    
    Parameters:
    -----------
    Constraints : numpy.ndarray or None
        NCons x 3 array containing [node_number, dof_number, value]
    dofs_per_node : int
        Number of degrees of freedom per node (positive integer)
    NCons : int
        Total number of Dirichlet boundary constraints (non-negative integer)
    NumNodes : int
        Total number of nodes in the mesh (positive integer)
    
    Returns:
    --------
    GlobalID : numpy.ndarray
        NumNodes x dofs_per_node matrix containing:
            Positive integers (1,2,3,...): equation numbers for unconstrained DOFs
            Negative integers (-1,-2,-3,...): constraint numbers for constrained DOFs
    eqn_num : int
        Total number of free equations (unconstrained DOFs)
    
    Example:
    --------
    >>> Constraints = np.array([[1, 1, 0.0], [1, 2, 0.0], [3, 1, 0.5]])
    >>> GlobalID, eqn_num = Create_ID_Matrix(Constraints, 2, 3, 5)
    """
    
    # ===================== INPUT VALIDATION =====================
    
    # Validate dofs_per_node
    if not isinstance(dofs_per_node, (int, np.integer)) or dofs_per_node < 1:
        raise ValueError('ERROR: dofs_per_node must be a positive integer')
    
    # Validate NCons
    if not isinstance(NCons, (int, np.integer)) or NCons < 0:
        raise ValueError('ERROR: NCons must be a non-negative integer')
    
    # Validate NumNodes
    if not isinstance(NumNodes, (int, np.integer)) or NumNodes < 1:
        raise ValueError('ERROR: NumNodes must be a positive integer')
    
    # Validate Constraints matrix dimensions
    if Constraints is None or len(Constraints) == 0:
        if NCons != 0:
            raise ValueError(f'ERROR: Constraints matrix is empty but NCons = {NCons}')
        Constraints = np.array([]).reshape(0, 3)  # Empty array with correct shape
    else:
        Constraints = np.array(Constraints)
        if Constraints.shape[1] != 3:
            raise ValueError('ERROR: Constraints matrix must have exactly 3 columns [node, dof, value]')
        if Constraints.shape[0] != NCons:
            raise ValueError(f'ERROR: Constraints has {Constraints.shape[0]} rows but NCons = {NCons}')
    
    # ===================== INITIALIZATION =====================
    
    # Initialize GlobalID matrix (0 = unprocessed)
    GlobalID = np.zeros((NumNodes, dofs_per_node), dtype=int)
    
    # ===================== PROCESS CONSTRAINTS =====================
    
    # Process each constraint
    for i in range(NCons):
        node_num = int(round(Constraints[i, 0]))
        dof_num = int(round(Constraints[i, 1]))
        
        # Validate node number (convert to 0-based indexing for Python)
        if node_num < 1 or node_num > NumNodes:
            raise ValueError(f'ERROR: Constraint {i+1} has invalid node number {node_num} '
                           f'(valid range: 1 to {NumNodes})')
        
        # Validate DOF number (convert to 0-based indexing for Python)
        if dof_num < 1 or dof_num > dofs_per_node:
            raise ValueError(f'ERROR: Constraint {i+1} has invalid DOF number {dof_num} '
                           f'(valid range: 1 to {dofs_per_node})')
        
        # Convert to 0-based indexing for internal storage
        node_idx = node_num - 1
        dof_idx = dof_num - 1
        
        # Check for duplicate constraint
        if GlobalID[node_idx, dof_idx] != 0:
            raise ValueError(f'ERROR: Duplicate constraint detected for node {node_num}, DOF {dof_num}')
        
        # Assign constraint number (negative: -1, -2, -3, ...)
        GlobalID[node_idx, dof_idx] = -(i + 1)
    
    # ===================== ASSIGN EQUATION NUMBERS =====================
    
    # Traverse all DOFs and assign positive equation numbers to unconstrained DOFs
    eqn_num = 0
    for node_i in range(NumNodes):
        for dof_j in range(dofs_per_node):
            if GlobalID[node_i, dof_j] == 0:  # Unconstrained DOF
                eqn_num += 1
                GlobalID[node_i, dof_j] = eqn_num
    
    return GlobalID, eqn_num