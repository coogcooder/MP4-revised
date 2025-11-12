import numpy as np

def ShapeFunctions(EleType, zeta):
    """
    Compute shape functions and their derivatives for various finite element types.
    
    This function evaluates shape functions N and their spatial derivatives DN at a
    given point in the reference element. Essential for FE assembly (stiffness matrix,
    mass matrix, load vectors).
    
    Parameters:
    -----------
    EleType : str
        Element type identifier:
        - 'L2': 2-node 1D linear element (-1 <= x <= +1)
        - 'L3': 3-node 1D quadratic element
        - 'Q4': 4-node bilinear quadrilateral (-1 <= x,y <= +1)
        - 'Q8': 8-node serendipity quadrilateral
        - 'T3': 3-node linear triangle (right-angled, h=b=1)
        - 'T6': 6-node quadratic triangle (right-angled, h=b=1)
        - 'B8': 8-node trilinear brick (-1 <= x,y,z <= +1)
        - 'TET4': 4-node linear tetrahedron (right-angled)
        - 'W6': 6-node wedge element
    
    zeta : array-like
        Coordinates of point in reference element
        - 1D elements: zeta = [xi]
        - 2D elements: zeta = [xi, eta]
        - 3D elements: zeta = [xi, eta, zeta]
    
    Returns:
    --------
    N : numpy.ndarray
        Shape functions evaluated at zeta point (size: num_nodes)
        N[i] = shape function value for node i
    
    DN : numpy.ndarray
        Derivatives of shape functions (size: num_nodes x spatial_dim)
        DN[i,j] = dN_i/d(zeta_j)
    
    Examples:
    ---------
    >>> # 2-node 1D element at xi = 0.5
    >>> N, DN = ShapeFunctions('L2', [0.5])
    >>> print(N)  # [0.25, 0.75]
    
    >>> # 4-node quadrilateral at (xi, eta) = (0, 0)
    >>> N, DN = ShapeFunctions('Q4', [0.0, 0.0])
    >>> print(N)  # [0.25, 0.25, 0.25, 0.25]
    """
    
    # Convert zeta to numpy array
    zeta = np.asarray(zeta, dtype=float)
    
    # Dispatch to appropriate element function
    if EleType == 'L2':
        return _L2(zeta)
    elif EleType == 'L3':
        return _L3(zeta)
    elif EleType == 'Q4':
        return _Q4(zeta)
    elif EleType == 'Q8':
        return _Q8(zeta)
    elif EleType == 'T3':
        return _T3(zeta)
    elif EleType == 'T6':
        return _T6(zeta)
    elif EleType == 'B8':
        return _B8(zeta)
    elif EleType == 'TET4':
        return _TET4(zeta)
    elif EleType == 'W6':
        return _W6(zeta)
    else:
        raise ValueError(f'Error in ShapeFunctions: Bad element type "{EleType}". '
                        f'Valid types: L2, L3, Q4, Q8, T3, T6, B8, TET4, W6')


def _L2(zeta):
    """2-node 1D linear element: -1 <= xi <= +1"""
    if zeta.size != 1:
        raise ValueError('Error in ShapeFunctions: L2 element requires 1D coordinate')
    
    xi = zeta[0]
    
    # Shape functions
    N = np.array([(1 - xi) / 2,
                  (1 + xi) / 2])
    
    # Derivatives: dN/dxi
    DN = np.array([[-0.5],
                   [0.5]])
    
    return N, DN


def _L3(zeta):
    """3-node 1D quadratic element"""
    if zeta.size != 1:
        raise ValueError('Error in ShapeFunctions: L3 element requires 1D coordinate')
    
    xi = zeta[0]
    
    # Shape functions (nodes at xi = -1, 0, +1)
    N = np.array([xi * (xi - 1) / 2,
                  1 - xi**2,
                  xi * (xi + 1) / 2])
    
    # Derivatives: dN/dxi
    DN = np.array([[(2*xi - 1) / 2],
                   [-2*xi],
                   [(2*xi + 1) / 2]])
    
    return N, DN


def _Q4(zeta):
    """4-node bilinear quadrilateral: -1 <= xi,eta <= +1"""
    if zeta.size != 2:
        raise ValueError('Error in ShapeFunctions: Q4 element requires 2D coordinates')
    
    xi, eta = zeta[0], zeta[1]
    
    # Shape functions (nodes at corners)
    N = np.array([(1 - xi) * (1 - eta) / 4,
                  (1 + xi) * (1 - eta) / 4,
                  (1 + xi) * (1 + eta) / 4,
                  (1 - xi) * (1 + eta) / 4])
    
    # Derivatives: dN/dxi, dN/deta
    DN = np.array([
        [-(1 - eta) / 4, -(1 - xi) / 4],
        [(1 - eta) / 4, -(1 + xi) / 4],
        [(1 + eta) / 4, (1 + xi) / 4],
        [-(1 + eta) / 4, (1 - xi) / 4]
    ])
    
    return N, DN


def _Q8(zeta):
    """8-node serendipity quadrilateral element"""
    if zeta.size != 2:
        raise ValueError('Error in ShapeFunctions: Q8 element requires 2D coordinates')
    
    xi, eta = zeta[0], zeta[1]
    
    # Corner nodes (1, 3, 5, 7 in local numbering: numbered counter-clockwise from bottom-left)
    # Midside nodes (2, 4, 6, 8)
    
    N = np.zeros(8)
    DN = np.zeros((8, 2))
    
    # Corner nodes
    for i in range(4):
        corner_xi = -1 if i < 2 else 1
        corner_eta = -1 if i % 2 == 0 else 1
        
        N[2*i] = (1 + xi*corner_xi) * (1 + eta*corner_eta) * (xi*corner_xi + eta*corner_eta - 1) / 4
        DN[2*i, 0] = corner_xi * (1 + eta*corner_eta) * (2*xi*corner_xi + eta*corner_eta) / 4
        DN[2*i, 1] = eta*corner_eta * (1 + xi*corner_xi) / 4 + \
                     corner_eta * (1 + xi*corner_xi) * (xi*corner_xi + eta*corner_eta - 1) / 4
    
    # Midside nodes
    # Node 2: xi=0, eta=-1
    N[1] = (1 - xi**2) * (1 - eta) / 2
    DN[1, 0] = -2*xi * (1 - eta) / 2
    DN[1, 1] = -(1 - xi**2) / 2
    
    # Node 4: xi=1, eta=0
    N[3] = (1 + xi) * (1 - eta**2) / 2
    DN[3, 0] = (1 - eta**2) / 2
    DN[3, 1] = (1 + xi) * (-2*eta) / 2
    
    # Node 6: xi=0, eta=1
    N[5] = (1 - xi**2) * (1 + eta) / 2
    DN[5, 0] = -2*xi * (1 + eta) / 2
    DN[5, 1] = (1 - xi**2) / 2
    
    # Node 8: xi=-1, eta=0
    N[7] = (1 - xi) * (1 - eta**2) / 2
    DN[7, 0] = -(1 - eta**2) / 2
    DN[7, 1] = (1 - xi) * (-2*eta) / 2
    
    return N, DN


def _T3(zeta):
    """3-node linear triangle: right-angled with h=b=1, nodes at (0,0), (1,0), (0,1)"""
    if zeta.size != 2:
        raise ValueError('Error in ShapeFunctions: T3 element requires 2D coordinates')
    
    x, y = zeta[0], zeta[1]
    
    # Shape functions (linear triangle using Cartesian coordinates)
    # Node 1: (0, 0)
    # Node 2: (1, 0)
    # Node 3: (0, 1)
    
    N = np.array([1 - x - y,
                  x,
                  y])
    
    # Derivatives: dN/dx, dN/dy
    DN = np.array([
        [-1, -1],
        [1, 0],
        [0, 1]
    ])
    
    return N, DN


def _T6(zeta):
    """6-node quadratic triangle: right-angled with h=b=1"""
    if zeta.size != 2:
        raise ValueError('Error in ShapeFunctions: T6 element requires 2D coordinates')
    
    x, y = zeta[0], zeta[1]
    
    # Nodes: 3 corners + 3 midedges
    # Corner nodes: (0,0), (1,0), (0,1)
    # Midedge nodes: (0.5,0), (0.5,0.5), (0,0.5)
    
    # Natural coordinates
    L1 = 1 - x - y  # Node 1
    L2 = x          # Node 2
    L3 = y          # Node 3
    
    # Shape functions
    N = np.array([
        L1 * (2*L1 - 1),        # Node 1 (corner)
        L2 * (2*L2 - 1),        # Node 2 (corner)
        L3 * (2*L3 - 1),        # Node 3 (corner)
        4 * L1 * L2,            # Node 4 (mid-edge 1-2)
        4 * L2 * L3,            # Node 5 (mid-edge 2-3)
        4 * L3 * L1             # Node 6 (mid-edge 3-1)
    ])
    
    # Derivatives
    DN = np.zeros((6, 2))
    
    # dN/dx for each node
    DN[0, 0] = -(2*L1 - 1) - 2*L1  # dL1/dx = -1
    DN[1, 0] = 4*L2 - 1
    DN[2, 0] = 0
    DN[3, 0] = 4*L1 - 4*L2
    DN[4, 0] = 4*L3
    DN[5, 0] = -4*L3
    
    # dN/dy for each node
    DN[0, 1] = -(2*L1 - 1) - 2*L1  # dL1/dy = -1
    DN[1, 1] = 0
    DN[2, 1] = 4*L3 - 1
    DN[3, 1] = -4*L2
    DN[4, 1] = 4*L2
    DN[5, 1] = 4*L1 - 4*L3
    
    return N, DN


def _B8(zeta):
    """8-node trilinear brick: -1 <= xi,eta,zeta <= +1"""
    if zeta.size != 3:
        raise ValueError('Error in ShapeFunctions: B8 element requires 3D coordinates')
    
    xi, eta, zeta_var = zeta[0], zeta[1], zeta[2]
    
    N = np.zeros(8)
    DN = np.zeros((8, 3))
    
    # 8 nodes at corners of unit cube
    corners = [
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
    ]
    
    for i, (cx, cy, cz) in enumerate(corners):
        N[i] = (1 + xi*cx) * (1 + eta*cy) * (1 + zeta_var*cz) / 8
        DN[i, 0] = cx * (1 + eta*cy) * (1 + zeta_var*cz) / 8
        DN[i, 1] = (1 + xi*cx) * cy * (1 + zeta_var*cz) / 8
        DN[i, 2] = (1 + xi*cx) * (1 + eta*cy) * cz / 8
    
    return N, DN


def _TET4(zeta):
    """4-node linear tetrahedron: right-angled at origin"""
    if zeta.size != 3:
        raise ValueError('Error in ShapeFunctions: TET4 element requires 3D coordinates')
    
    x, y, z = zeta[0], zeta[1], zeta[2]
    
    # Nodes at: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    # Natural coordinates L1, L2, L3, L4 where L1+L2+L3+L4=1
    # L1 = 1-x-y-z, L2 = x, L3 = y, L4 = z
    
    L1 = 1 - x - y - z
    L2 = x
    L3 = y
    L4 = z
    
    N = np.array([L1, L2, L3, L4])
    
    # Derivatives with respect to x, y, z
    DN = np.array([
        [-1, -1, -1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    return N, DN


def _W6(zeta):
    """6-node wedge (prism) element: triangular base extruded in z"""
    if zeta.size != 3:
        raise ValueError('Error in ShapeFunctions: W6 element requires 3D coordinates')
    
    x, y, z = zeta[0], zeta[1], zeta[2]
    
    # Triangular coordinates in x-y plane (like T3)
    L1 = 1 - x - y
    L2 = x
    L3 = y
    
    # Linear in z direction: z ranges from 0 to 1 (or -1 to +1 depending on convention)
    # Assuming z in [0, 1]
    Nz_bot = 1 - z  # Bottom face (z=0)
    Nz_top = z      # Top face (z=1)
    
    # 6 nodes: 3 on bottom triangle, 3 on top triangle
    N = np.array([
        L1 * Nz_bot,  # Node 1: (0,0,0)
        L2 * Nz_bot,  # Node 2: (1,0,0)
        L3 * Nz_bot,  # Node 3: (0,1,0)
        L1 * Nz_top,  # Node 4: (0,0,1)
        L2 * Nz_top,  # Node 5: (1,0,1)
        L3 * Nz_top   # Node 6: (0,1,1)
    ])
    
    # Derivatives: dN/dx, dN/dy, dN/dz
    DN = np.zeros((6, 3))
    
    # Derivatives with respect to x (dL1/dx=-1, dL2/dx=1, dL3/dx=0)
    DN[0, 0] = -Nz_bot
    DN[1, 0] = Nz_bot
    DN[2, 0] = 0
    DN[3, 0] = -Nz_top
    DN[4, 0] = Nz_top
    DN[5, 0] = 0
    
    # Derivatives with respect to y (dL1/dy=-1, dL2/dy=0, dL3/dy=1)
    DN[0, 1] = -Nz_bot
    DN[1, 1] = 0
    DN[2, 1] = Nz_bot
    DN[3, 1] = -Nz_top
    DN[4, 1] = 0
    DN[5, 1] = Nz_top
    
    # Derivatives with respect to z (dNz_bot/dz=-1, dNz_top/dz=1)
    DN[0, 2] = -L1
    DN[1, 2] = -L2
    DN[2, 2] = -L3
    DN[3, 2] = L1
    DN[4, 2] = L2
    DN[5, 2] = L3
    
    return N, DN