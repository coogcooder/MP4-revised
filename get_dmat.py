# get_dmat.py
import numpy as np

def Get_DMat(diffusivity_function, x, dim=None):
    """
    Returns the diffusivity tensor DMat at spatial point x.

    Supported 'diffusivity_function' layouts:
      - {'type':'isotropic', 'value': float}
      - {'type':'tensor', 'value': (dxd) np.array}
      - {'type':'callable', 'func': callable(x)->scalar or (dxd) array}
    If isotropic scalar is provided and 'dim' is not None, returns value*I_dim.
    """
    if diffusivity_function is None:
        raise ValueError("diffusivity_function must be provided")

    ftype = diffusivity_function.get('type', '').lower()

    if ftype == 'isotropic':
        val = float(diffusivity_function['value'])
        if dim is None:
            # Fall back to 1D if dim not supplied
            return np.array([[val]], dtype=float)
        return val * np.eye(dim, dtype=float)

    elif ftype == 'tensor':
        D = np.array(diffusivity_function['value'], dtype=float)
        return D

    elif ftype == 'callable':
        func = diffusivity_function['func']
        out = func(np.asarray(x, dtype=float))
        out = np.array(out, dtype=float)
        if out.ndim == 0:
            # scalar -> make isotropic (needs dim)
            if dim is None:
                return np.array([[float(out)]], dtype=float)
            return float(out) * np.eye(dim, dtype=float)
        return out

    else:
        raise ValueError(f"Unsupported diffusivity_function type: {ftype}")
