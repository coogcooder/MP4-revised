# get_volumetric_source.py
import numpy as np

def Get_VolumetricSource(load_type, x):
    """
    Returns volumetric source f at spatial point x.

    Supported 'load_type' layouts:
      - {'type':'constant', 'value': float}
      - {'type':'callable', 'func': callable(x)->float}
    """
    if load_type is None:
        return 0.0

    ltype = load_type.get('type', '').lower()

    if ltype == 'constant':
        return float(load_type['value'])

    elif ltype == 'callable':
        func = load_type['func']
        return float(func(np.asarray(x, dtype=float)))

    else:
        raise ValueError(f"Unsupported load_type type: {ltype}")
