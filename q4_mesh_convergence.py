#!/usr/bin/env python3
"""
Q4 Element Convergence Study with Analytical Solution
======================================================
Solves 2D Poisson equation on unit square with homogeneous Dirichlet BCs
Compares FE solution with analytical solution and performs convergence analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add paths for imports
sys.path.insert(0, '/mnt/user-data/outputs')
sys.path.insert(0, '/mnt/user-data/uploads')

from driver_steady_diffusion import Driver_Steady_Diffusion
from shape_functions import ShapeFunctions
from gauss_quadrature import GaussPoints


def analytical_solution(x, y):
    """Analytical solution: u(x,y) = sin(πx)sin(πy)"""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def analytical_gradient(x, y):
    """Gradient of analytical solution"""
    dudx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    dudy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    return dudx, dudy


def source_function(x):
    """Source term: f = 2π²sin(πx)sin(πy)"""
    return 2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def generate_q4_mesh(nx, ny):
    """Generate Q4 mesh for unit square [0,1]×[0,1]"""
    # Create nodes
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    
    NumNodes = (nx + 1) * (ny + 1)
    Coord = np.zeros((NumNodes, 2))
    
    # Node numbering: row by row, bottom to top
    idx = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            Coord[idx] = [x[i], y[j]]
            idx += 1
    
    # Create connectivity (1-based indexing)
    Nele = nx * ny
    Connectivity = np.zeros((Nele, 4), dtype=int)
    
    elem = 0
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i + 1  # Bottom-left (1-based)
            n2 = n1 + 1                 # Bottom-right
            n3 = n1 + nx + 2           # Top-right
            n4 = n1 + nx + 1           # Top-left
            Connectivity[elem] = [n1, n2, n3, n4]
            elem += 1
    
    # Identify boundary nodes for homogeneous Dirichlet BCs
    boundary_nodes = []
    for idx in range(NumNodes):
        x, y = Coord[idx]
        if np.isclose(x, 0) or np.isclose(x, 1) or np.isclose(y, 0) or np.isclose(y, 1):
            boundary_nodes.append(idx + 1)  # 1-based
    
    # Create constraints (u = 0 on boundary)
    Constraints = np.array([[node, 1, 0.0] for node in boundary_nodes])
    NCons = len(boundary_nodes)
    
    return Coord, Connectivity, Constraints, NCons, Nele, NumNodes


def compute_errors(U_fe, Coord, Connectivity, n_gauss=3):
    """Compute L2 and H1 errors using Gauss quadrature"""
    L2_error_squared = 0.0
    H1_error_squared = 0.0
    
    # Get Gauss points and weights
    r, w = GaussPoints(2, 'Q4', n_gauss)
    
    for elem in Connectivity:
        # Get element nodes (convert to 0-based)
        nodes = elem - 1
        xe = Coord[nodes]
        ue_fe = U_fe[nodes].flatten()
        
        # Integrate over element
        for igp in range(len(w)):
            # Shape functions and derivatives at Gauss point
            N, dN_dxi = ShapeFunctions('Q4', r[igp])
            N = N.flatten()
            
            # Jacobian
            if dN_dxi.shape == (4, 2):
                J = xe.T @ dN_dxi
                dN_dx = dN_dxi @ np.linalg.inv(J)
            else:
                J = dN_dxi @ xe
                dN_dx = (np.linalg.inv(J).T @ dN_dxi).T
            
            detJ = np.linalg.det(J)
            
            # Physical coordinates
            x_phys = N @ xe
            
            # FE solution and gradient at Gauss point
            u_fe_gp = N @ ue_fe
            grad_u_fe = dN_dx.T @ ue_fe
            
            # Analytical solution and gradient
            u_exact = analytical_solution(x_phys[0], x_phys[1])
            grad_u_exact = analytical_gradient(x_phys[0], x_phys[1])
            
            # Accumulate errors
            L2_error_squared += (u_fe_gp - u_exact)**2 * detJ * w[igp]
            grad_error = grad_u_fe - np.array(grad_u_exact)
            H1_error_squared += np.dot(grad_error, grad_error) * detJ * w[igp]
    
    L2_error = np.sqrt(L2_error_squared)
    H1_seminorm_error = np.sqrt(H1_error_squared)
    
    return L2_error, H1_seminorm_error


def save_mesh_data(Coord, Connectivity, Constraints, h, folder='mesh_data'):
    """Save mesh data to text files"""
    os.makedirs(folder, exist_ok=True)
    
    h_str = f"{h:.4f}".replace('.', '_')
    np.savetxt(f'{folder}/coord_h{h_str}.txt', Coord, fmt='%.8e', 
               header='x y coordinates')
    np.savetxt(f'{folder}/connectivity_h{h_str}.txt', Connectivity, fmt='%d',
               header='Q4 element connectivity (1-based)')
    np.savetxt(f'{folder}/constraints_h{h_str}.txt', Constraints, fmt='%.8e',
               header='node dof value')
    print(f"  Saved mesh data for h={h:.4f}")


def main():
    print("="*70)
    print("Q4 ELEMENT CONVERGENCE STUDY")
    print("Problem: -∇²u = f on [0,1]×[0,1], u=0 on boundary")
    print("Analytical solution: u(x,y) = sin(πx)sin(πy)")
    print("="*70)
    
    # Mesh sizes to test (hierarchical refinement)
    mesh_sizes = [2, 4, 8, 16, 32]
    h_values = []
    L2_errors = []
    H1_errors = []
    
    print("\nPerforming convergence study...")
    print("-"*50)
    
    # Problem parameters
    dim = 2
    dofs_per_node = 1
    EleType = 'Q4'
    NGPTS = 3  # Use 3x3 Gauss quadrature for better accuracy
    diffusion_function = {'type': 'isotropic', 'value': 1.0}
    load_type = {'type': 'callable', 'func': source_function}
    
    # Store finest mesh solution for visualization
    finest_solution = None
    finest_coord = None
    finest_connectivity = None
    finest_nx = None
    
    for n in mesh_sizes:
        h = 1.0 / n
        h_values.append(h)
        
        print(f"\nMesh {n}×{n} (h={h:.4f}):")
        
        # Generate mesh
        Coord, Connectivity, Constraints, NCons, Nele, NumNodes = generate_q4_mesh(n, n)
        
        # Save mesh data
        save_mesh_data(Coord, Connectivity, Constraints, h)
        
        # Solve FE problem
        U = Driver_Steady_Diffusion(
            Connectivity, Constraints, Coord, diffusion_function,
            dim, dofs_per_node, EleType, load_type, NCons,
            Nele, NGPTS, NumNodes
        )
        
        # Compute errors
        L2_error, H1_error = compute_errors(U, Coord, Connectivity, n_gauss=3)
        L2_errors.append(L2_error)
        H1_errors.append(H1_error)
        
        print(f"  Nodes: {NumNodes}, Elements: {Nele}")
        print(f"  L2 error: {L2_error:.6e}")
        print(f"  H1 seminorm error: {H1_error:.6e}")
        
        # Store finest mesh for visualization
        if n == mesh_sizes[-1]:
            finest_solution = U
            finest_coord = Coord
            finest_connectivity = Connectivity
            finest_nx = n
    
  # ------------------------------------------------------------
    # Convergence rates and plot against -log(h)
    # ------------------------------------------------------------
    h_values = np.array(h_values)
    L2_errors = np.array(L2_errors)
    H1_errors = np.array(H1_errors)

    # x = -log(h); y = log(error)
    x = -np.log(h_values)
    yL2 = np.log(L2_errors)
    yH1 = np.log(H1_errors)

    # Linear regression in this space: y = a + m * x
    mL2, aL2 = np.polyfit(x, yL2, 1)  # slope should be about -2
    mH1, aH1 = np.polyfit(x, yH1, 1)  # slope should be about -1

    print("\n" + "="*50)
    print("CONVERGENCE RATES (plotted vs -log h):")
    print(f"  L2 slope: {mL2:.3f}  (theory: -2.0 for Q4)")
    print(f"  H1 slope: {mH1:.3f}  (theory: -1.0 for Q4)")
    print("="*50)

    # Plot log(error) vs -log(h) (no log axes needed; we've already transformed)
    fig1 = plt.figure(figsize=(10, 6))
    ax = fig1.add_subplot(111)

    ax.plot(x, yL2, 'bo-', linewidth=2, markersize=8, label=f"L²: slope={mL2:.2f}")
    ax.plot(x, yH1, 'rs-', linewidth=2, markersize=8, label=f"H¹: slope={mH1:.2f}")

    # Reference lines with slopes -2 and -1 through the last data point for clarity
    xref = np.linspace(x.min(), x.max(), 100)
    # anchor at the last L2 point:
    x0, y0 = x[-1], yL2[-1]
    refL2 = y0 + (-2.0) * (xref - x0)
    # anchor at the last H1 point:
    x1, y1 = x[-1], yH1[-1]
    refH1 = y1 + (-1.0) * (xref - x1)

    ax.plot(xref, refL2, 'b--', alpha=0.5, label="slope -2 reference")
    ax.plot(xref, refH1, 'r--', alpha=0.5, label="slope -1 reference")

    ax.set_xlabel(r"$-\log(h)$", fontsize=12)
    ax.set_ylabel(r"$\log(\mathrm{error})$", fontsize=12)
    ax.set_title("Convergence Study: Q4 Elements (log(error) vs -log(h))", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig('q4_convergence.png', dpi=150)
    print("\nConvergence plot saved as 'q4_convergence.png'")
    
    # 3D Visualization
    print("\nGenerating 3D visualizations...")
    
    fig2 = plt.figure(figsize=(14, 6))
    
    # Prepare data for 3D plots
    nx = finest_nx
    X = finest_coord[:, 0].reshape(nx+1, nx+1)
    Y = finest_coord[:, 1].reshape(nx+1, nx+1)
    U_fe_grid = finest_solution.reshape(nx+1, nx+1)
    U_exact_grid = analytical_solution(X, Y)
    
    # FE Solution
    ax1 = fig2.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, U_fe_grid, cmap='viridis', 
                             edgecolor='none', alpha=0.9)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    ax1.set_title(f'FE Solution ({nx}×{nx} mesh)')
    fig2.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # Analytical Solution
    ax2 = fig2.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, U_exact_grid, cmap='plasma', 
                             edgecolor='none', alpha=0.9)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u(x,y)')
    ax2.set_title('Analytical Solution')
    fig2.colorbar(surf2, ax=ax2, shrink=0.5)
    
    plt.suptitle('Q4 Finite Element vs Analytical Solution', fontsize=14)
    plt.tight_layout()
    plt.savefig('q4_3d_comparison.png', dpi=150)
    print("3D visualization saved as 'q4_3d_comparison.png'")
    
    # Error distribution plot
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    
    error_grid = np.abs(U_fe_grid - U_exact_grid)
    im = ax3.contourf(X, Y, error_grid, levels=20, cmap='hot')
    ax3.contour(X, Y, error_grid, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(im, ax=ax3)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title(f'Absolute Error Distribution ({nx}×{nx} mesh)')
    ax3.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('q4_error_distribution.png', dpi=150)
    print("Error distribution saved as 'q4_error_distribution.png'")
    
    # Display plots
    plt.show()
    
    print("\n" + "="*70)
    print("CONVERGENCE STUDY COMPLETED SUCCESSFULLY!")
    print("Generated files:")
    print("  - Mesh data in 'mesh_data/' folder")
    print("  - q4_convergence.png (log-log convergence plot)")
    print("  - q4_3d_comparison.png (3D solution visualization)")
    print("  - q4_error_distribution.png (error distribution)")
    print("="*70)


if __name__ == "__main__":
    main()