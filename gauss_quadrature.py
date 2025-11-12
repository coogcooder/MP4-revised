"""
Automated Gaussian Quadrature Point Generator
==============================================
Generates Gauss quadrature points and weights for all supported element types
across 1D, 2D, and 3D with comprehensive validation and file output.
"""

import numpy as np
from datetime import datetime
import os


# ============================================================================
# MAIN DISPATCHER
# ============================================================================

def GaussPoints(dim, EleType, NGPTS):
    """
    Main dispatcher for Gaussian quadrature points and weights.
    
    Parameters:
        dim (int): Dimension (1, 2, or 3)
        EleType (str): Element type
        NGPTS (int): Number of Gauss points
        
    Returns:
        r (ndarray): Quadrature point locations (n_points × dim)
        w (ndarray): Weights (n_points)
    """
    if dim == 1:
        return Gauss_1D(EleType, NGPTS)
    elif dim == 2:
        return Gauss_2D(EleType, NGPTS)
    elif dim == 3:
        return Gauss_3D(EleType, NGPTS)
    else:
        raise ValueError(f"Invalid dimension: {dim}. Must be 1, 2, or 3.")


# ============================================================================
# 1D QUADRATURE
# ============================================================================

def Gauss_1D(EleType, NGPTS):
    """1D Gauss quadrature for line element [-1, 1]"""
    # Convert to uppercase for comparison
    EleType = EleType.upper()
    
    # Check if it's a valid 1D element type
    if EleType not in ['L', 'L2', 'L3']:
        raise ValueError(f"Invalid 1D element: {EleType}. Valid types are: L, L2, L3")
    
    if not 1 <= NGPTS <= 5:
        raise ValueError(f"NGPTS must be 1-5 for 1D, got {NGPTS}")
    
    points, weights = np.polynomial.legendre.leggauss(NGPTS)
    return points.reshape(-1, 1), weights


# ============================================================================
# 2D QUADRATURE
# ============================================================================

def Gauss_2D(EleType, NGPTS):
    """2D Gauss quadrature dispatcher"""
    EleType = EleType.upper()
    if EleType in ['T3', 'T4', 'T6']:
        return Gauss_Triangle(NGPTS)
    elif EleType in ['Q4', 'Q5', 'Q8', 'Q9']:
        return Gauss_Quadrilateral(NGPTS)
    else:
        raise ValueError(f"Invalid 2D element: {EleType}")


def Gauss_Triangle(NGPTS):
    """Triangle quadrature - domain: (0,0), (1,0), (0,1); measure: 0.5"""
    if NGPTS == 1:
        r = np.array([[1/3, 1/3]])
        w = np.array([0.5])
    elif NGPTS == 3:
        r = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
        w = np.array([1/6, 1/6, 1/6])
    elif NGPTS == 4:
        r = np.array([[1/3, 1/3], [1/5, 1/5], [3/5, 1/5], [1/5, 3/5]])
        w = np.array([-27/96, 25/96, 25/96, 25/96])
    elif NGPTS == 7:
        r = np.array([
            [1/3, 1/3],
            [0.059715871789770, 0.059715871789770],
            [0.797426985353087, 0.059715871789770],
            [0.059715871789770, 0.797426985353087],
            [0.470142064105115, 0.470142064105115],
            [0.101286507323456, 0.470142064105115],
            [0.470142064105115, 0.101286507323456]
        ])
        w = np.array([0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506,
                      0.125939180544827, 0.125939180544827, 0.125939180544827]) / 2
    else:
        raise ValueError(f"Triangle NGPTS must be 1, 3, 4, or 7")
    return r, w


def Gauss_Quadrilateral(NGPTS):
    """Quadrilateral tensor product - domain: [-1,1]×[-1,1]; measure: 4"""
    if NGPTS not in [2, 3, 4, 5]:
        raise ValueError(f"Quad NGPTS must be 2, 3, 4, or 5")
    
    xi_1d, w_1d = np.polynomial.legendre.leggauss(NGPTS)
    total = NGPTS * NGPTS
    r = np.zeros((total, 2))
    w = np.zeros(total)
    
    idx = 0
    for i in range(NGPTS):
        for j in range(NGPTS):
            r[idx] = [xi_1d[i], xi_1d[j]]
            w[idx] = w_1d[i] * w_1d[j]
            idx += 1
    return r, w


# ============================================================================
# 3D QUADRATURE
# ============================================================================

def Gauss_3D(EleType, NGPTS):
    """3D Gauss quadrature dispatcher"""
    EleType = EleType.upper()
    if EleType in ['B8', 'B27']:
        return Gauss_Brick(NGPTS)
    elif EleType in ['TET4', 'TET10']:
        return Gauss_Tetrahedron(NGPTS)
    elif EleType in ['W6', 'W15']:
        return Gauss_Wedge(NGPTS)
    else:
        raise ValueError(f"Invalid 3D element: {EleType}")


def Gauss_Brick(NGPTS):
    """Brick tensor product - domain: [-1,1]³; measure: 8"""
    if not 1 <= NGPTS <= 5:
        raise ValueError(f"Brick NGPTS should be 1-5")
    
    xi_1d, w_1d = np.polynomial.legendre.leggauss(NGPTS)
    total = NGPTS ** 3
    r = np.zeros((total, 3))
    w = np.zeros(total)
    
    idx = 0
    for i in range(NGPTS):
        for j in range(NGPTS):
            for k in range(NGPTS):
                r[idx] = [xi_1d[i], xi_1d[j], xi_1d[k]]
                w[idx] = w_1d[i] * w_1d[j] * w_1d[k]
                idx += 1
    return r, w


def Gauss_Tetrahedron(NGPTS):
    """Tetrahedron - domain: (0,0,0), (1,0,0), (0,1,0), (0,0,1); measure: 1/6"""
    if NGPTS == 1:
        r = np.array([[0.25, 0.25, 0.25]])
        w = np.array([1/6])
    elif NGPTS == 4:
        a, b = 0.585410196624969, 0.138196601125011
        r = np.array([[b, b, b], [a, b, b], [b, a, b], [b, b, a]])
        w = np.array([1/24, 1/24, 1/24, 1/24])
    elif NGPTS == 5:
        r = np.array([[0.25, 0.25, 0.25], [1/6, 1/6, 1/6],
                      [0.5, 1/6, 1/6], [1/6, 0.5, 1/6], [1/6, 1/6, 0.5]])
        w = np.array([-2/15, 3/40, 3/40, 3/40, 3/40]) / 6
    else:
        raise ValueError(f"Tet NGPTS must be 1, 4, or 5")
    return r, w


def Gauss_Wedge(NGPTS):
    """Wedge tensor product (triangle × line) - measure: 1.0"""
    if NGPTS not in [1, 3, 4, 7]:
        raise ValueError(f"Wedge NGPTS must be 1, 3, 4, or 7")
    
    # Triangle in xi-eta plane
    r_tri, w_tri = Gauss_Triangle(NGPTS)
    n_tri = len(w_tri)
    
    # Line in zeta direction (2 points default)
    xi_line, w_line = np.polynomial.legendre.leggauss(2)
    
    # Tensor product
    total = n_tri * 2
    r = np.zeros((total, 3))
    w = np.zeros(total)
    
    idx = 0
    for i in range(n_tri):
        for j in range(2):
            r[idx] = [r_tri[i, 0], r_tri[i, 1], xi_line[j]]
            w[idx] = w_tri[i] * w_line[j]
            idx += 1
    return r, w


# ============================================================================
# VALIDATION
# ============================================================================

def validate_weights(dim, EleType, w):
    """
    Validate that weights sum to element measure.
    
    Returns:
        is_valid (bool): True if validation passes
        expected (float): Expected measure
        actual (float): Actual sum of weights
    """
    # Expected measures for each element type
    measures = {
        'L': 2.0,                          # 1D line
        'T3': 0.5, 'T4': 0.5, 'T6': 0.5,  # 2D triangles
        'Q4': 4.0, 'Q5': 4.0, 'Q8': 4.0, 'Q9': 4.0,  # 2D quads
        'B8': 8.0, 'B27': 8.0,            # 3D bricks
        'TET4': 1/6, 'TET10': 1/6,        # 3D tetrahedra
        'W6': 1.0, 'W15': 1.0             # 3D wedges
    }
    
    expected = measures[EleType.upper()]
    actual = np.sum(w)
    is_valid = abs(actual - expected) < 1e-10
    
    return is_valid, expected, actual


# ============================================================================
# FILE OUTPUT
# ============================================================================

def save_results(dim, EleType, NGPTS, r, w, output_dir='GaussQuadrature_Results'):
    """
    Save quadrature results to text file with validation.
    
    Parameters:
        dim (int): Dimension
        EleType (str): Element type
        NGPTS (int): Number of Gauss points
        r (ndarray): Quadrature points
        w (ndarray): Weights
        output_dir (str): Output directory name
        
    Returns:
        filepath (str): Path to saved file
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate filename
    filename = f"{dim}D_{EleType}_{NGPTS}pts.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Validate weights
    is_valid, expected, actual = validate_weights(dim, EleType, w)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*70 + "\n")
        f.write("GAUSSIAN QUADRATURE RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"Generated:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dimension:        {dim}D\n")
        f.write(f"Element Type:     {EleType}\n")
        f.write(f"Requested Points: {NGPTS}\n")
        f.write(f"Actual Points:    {len(w)}\n")
        f.write("="*70 + "\n\n")
        
        # Quadrature points
        f.write("QUADRATURE POINTS (r):\n")
        f.write("-"*70 + "\n")
        
        if dim == 1:
            f.write(f"{'Point':<8} {'xi':<20}\n")
            f.write("-"*30 + "\n")
            for i in range(len(w)):
                f.write(f"{i+1:<8} {r[i,0]:20.15f}\n")
        elif dim == 2:
            f.write(f"{'Point':<8} {'xi':<20} {'eta':<20}\n")
            f.write("-"*50 + "\n")
            for i in range(len(w)):
                f.write(f"{i+1:<8} {r[i,0]:20.15f} {r[i,1]:20.15f}\n")
        else:  # dim == 3
            f.write(f"{'Point':<8} {'xi':<20} {'eta':<20} {'zeta':<20}\n")
            f.write("-"*70 + "\n")
            for i in range(len(w)):
                f.write(f"{i+1:<8} {r[i,0]:20.15f} {r[i,1]:20.15f} {r[i,2]:20.15f}\n")
        
        # Weights
        f.write("\n" + "="*70 + "\n")
        f.write("WEIGHTS (w):\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Point':<8} {'Weight':<25}\n")
        f.write("-"*35 + "\n")
        for i in range(len(w)):
            f.write(f"{i+1:<8} {w[i]:25.20f}\n")
        
        # Validation
        f.write("\n" + "="*70 + "\n")
        f.write("VALIDATION CHECK\n")
        f.write("="*70 + "\n")
        f.write(f"Expected measure: {expected:.20f}\n")
        f.write(f"Sum of weights:   {actual:.20f}\n")
        f.write(f"Difference:       {abs(actual - expected):.2e}\n")
        f.write(f"Status:           {'[PASS] ✓' if is_valid else '[FAIL] ✗'}\n")
        f.write("="*70 + "\n")
    
    return filepath, is_valid


# ============================================================================
# AUTOMATED BATCH PROCESSING
# ============================================================================

def run_all_configurations():
    """
    Run all valid combinations of dimension, element type, and Gauss points.
    Saves results to individual files and creates a summary report.
    """
    # Define all valid configurations
    # Format: dimension -> {element_type: [valid_NGPTS_list]}
    configurations = {
        1: {
            'L': [1, 2, 3, 4, 5]
        },
        2: {
            'T3': [1, 3, 4, 7],
            'T4': [1, 3, 4, 7],
            'T6': [1, 3, 4, 7],
            'Q4': [2, 3, 4, 5],
            'Q5': [2, 3, 4, 5],
            'Q8': [2, 3, 4, 5],
            'Q9': [2, 3, 4, 5]
        },
        3: {
            'B8': [2, 3, 4],
            'B27': [2, 3, 4],
            'TET4': [1, 4, 5],
            'TET10': [1, 4, 5],
            'W6': [1, 3, 4, 7],
            'W15': [1, 3, 4, 7]
        }
    }
    
    # Initialize summary tracking
    results_summary = []
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    print("\n" + "="*70)
    print("AUTOMATED GAUSSIAN QUADRATURE GENERATOR")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Loop through all configurations
    for dim in sorted(configurations.keys()):
        print(f"\n{'='*70}")
        print(f"Processing {dim}D Elements")
        print(f"{'='*70}")
        
        for ele_type in sorted(configurations[dim].keys()):
            print(f"\n  Element: {ele_type}")
            
            for ngpts in configurations[dim][ele_type]:
                total_tests += 1
                
                try:
                    # Generate quadrature
                    r, w = GaussPoints(dim, ele_type, ngpts)
                    
                    # Save results
                    filepath, is_valid = save_results(dim, ele_type, ngpts, r, w)
                    
                    # Track results
                    status = "PASS" if is_valid else "FAIL"
                    if is_valid:
                        passed_tests += 1
                    else:
                        failed_tests += 1
                    
                    results_summary.append({
                        'dim': dim,
                        'element': ele_type,
                        'ngpts': ngpts,
                        'actual_points': len(w),
                        'status': status,
                        'file': os.path.basename(filepath)
                    })
                    
                    print(f"    NGPTS={ngpts} ({len(w)} points): [{status}] -> {os.path.basename(filepath)}")
                    
                except Exception as e:
                    failed_tests += 1
                    results_summary.append({
                        'dim': dim,
                        'element': ele_type,
                        'ngpts': ngpts,
                        'actual_points': 'ERROR',
                        'status': 'ERROR',
                        'file': str(e)
                    })
                    print(f"    NGPTS={ngpts}: [ERROR] {str(e)}")
    
    # Save summary report
    save_summary_report(results_summary, total_tests, passed_tests, failed_tests)
    
    # Print final summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Total tests:  {total_tests}")
    print(f"Passed:       {passed_tests}")
    print(f"Failed:       {failed_tests}")
    print(f"Success rate: {100*passed_tests/total_tests:.1f}%")
    print(f"\nResults saved in: GaussQuadrature_Results/")
    print(f"Summary report:   GaussQuadrature_Results/SUMMARY_REPORT.txt")
    print("="*70 + "\n")


def save_summary_report(results_summary, total, passed, failed):
    """Save comprehensive summary report of all tests."""
    
    filepath = os.path.join('GaussQuadrature_Results', 'SUMMARY_REPORT.txt')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("GAUSSIAN QUADRATURE - COMPREHENSIVE TEST SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Tests: {total}\n")
        f.write(f"Passed:      {passed}\n")
        f.write(f"Failed:      {failed}\n")
        f.write(f"Success:     {100*passed/total:.2f}%\n")
        f.write("="*80 + "\n\n")
        
        # Detailed results table
        f.write("DETAILED RESULTS:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Dim':<5} {'Element':<10} {'NGPTS':<8} {'Points':<10} {'Status':<10} {'File':<35}\n")
        f.write("-"*80 + "\n")
        
        for result in results_summary:
            f.write(f"{result['dim']:<5} "
                   f"{result['element']:<10} "
                   f"{result['ngpts']:<8} "
                   f"{str(result['actual_points']):<10} "
                   f"{result['status']:<10} "
                   f"{result['file']:<35}\n")
        
        f.write("-"*80 + "\n")
        
        # Summary by dimension
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY BY DIMENSION:\n")
        f.write("="*80 + "\n")
        
        for dim in [1, 2, 3]:
            dim_results = [r for r in results_summary if r['dim'] == dim]
            dim_passed = sum(1 for r in dim_results if r['status'] == 'PASS')
            dim_total = len(dim_results)
            f.write(f"\n{dim}D Elements: {dim_passed}/{dim_total} passed "
                   f"({100*dim_passed/dim_total:.1f}%)\n")
            
            # Group by element type
            elements = set(r['element'] for r in dim_results)
            for elem in sorted(elements):
                elem_results = [r for r in dim_results if r['element'] == elem]
                elem_passed = sum(1 for r in elem_results if r['status'] == 'PASS')
                f.write(f"  {elem:<10}: {elem_passed}/{len(elem_results)} tests passed\n")
        
        f.write("\n" + "="*80 + "\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_all_configurations()