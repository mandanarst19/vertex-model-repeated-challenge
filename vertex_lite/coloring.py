
import numpy as np
from matplotlib import cm

from matplotlib.colors import rgb2hex,ListedColormap
import seaborn as sns

## 
def definecolors2(cells, property_name='parent_group', cmap_name='hls'):
    """
    Assign colors to cells based on a DISCRETE specified property.
    Colors are assigned consistently for each unique property value.
    First color is reserved for the background (0).
    Parameters:
        cells (Cells): The cells object containing a properties dictionary.
        property_name (str): The property key to use for colouring (default 'parent_group').
        cmap_name (str): The name of a matplotlib colormap (default 'hls').

    Returns:
        numpy.ndarray: An array of hex color codes corresponding to the property values.
    """
   
    if property_name not in cells.properties:
        cells.properties[property_name] = np.zeros(len(cells), dtype=int)
    # Get the property array (should be an array of integers like 0,1,2,..)
    prop_values = cells.properties[property_name]
    unique_groups = np.unique(prop_values)
    if 0 not in unique_groups:
        unique_groups = np.insert(unique_groups, 0, 0)
    # Find unique groups and sort for consistency.
    
    
    n_groups = len(unique_groups)
    cmap_size= max(n_groups, 10) # Ensure cmap_size is at least 10 for consistency.
    # If a seaborn palette is specified, convert it into a ListedColormap.
    if cmap_name.lower() in ['hls', 'husl', 'deep', 'muted', 'pastel', 'bright', 'dark']:
        palette = sns.color_palette(cmap_name, cmap_size)
        cmap = ListedColormap([rgb2hex(color) for color in palette])
    else:
        cmap = cm.get_cmap(cmap_name, cmap_size)
    
    # Create a dictionary mapping each group to a hex color.
    color_dict = {group: rgb2hex(cmap(i)) for i, group in enumerate(unique_groups)}
    
    # Assign a color to each cell based on its property value.
    colors = np.array([color_dict[val] for val in prop_values])
    return colors

def values_to_hex_colors2(values, cmap_name='Reds', fixed_vmin=3.72, fixed_vmax=3.9, limit_colors=True):

    # I chose vmax to be 1.10 since according to literature and experimental data it seemed like a sensible upper limit.
    # This is a bit arbitrary, but it is a good starting point, might be worth experimenting along these lines.
    """
    Map CONTINUOUS values to hex colors using a colormap.
    Uses a fixed minimum value of fixed_vmin (default 1) so that any value less than
    fixed_vmin returns white (#ffffff). If fixed_vmax is not provided, it uses the maximum
    among the values that are >= fixed_vmin.
    """
    from matplotlib.colors import Normalize, rgb2hex, TwoSlopeNorm

    if cmap_name == 'custom':
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        norm = TwoSlopeNorm(vmin=fixed_vmin, vcenter=3.81, vmax=fixed_vmax)
    else:
        norm = Normalize(vmin=fixed_vmin, vmax=fixed_vmax)
        cmap = cm.get_cmap(cmap_name)

    hex_colors = []
    for val in values:
        try:
            v = float(val)
        except Exception:
            # Could not convert → treat as invalid
            hex_colors.append("#00ffff")
            continue
        if limit_colors:
            # Now `v` is guaranteed to be a float.
            if np.isnan(v) or (v < fixed_vmin and cmap_name != 'custom'):
                hex_colors.append("#00ffff")
            elif v > fixed_vmax and cmap_name != 'custom':
                hex_colors.append("#ff0000")
            else:
                color = cmap(norm(v))
                hex_colors.append(rgb2hex(color))
        elif np.isnan(v):
            hex_colors.append("#00ffff")
        else:
            # Saturate values outside the normalization range at the colormap limits.
            bounded_v = np.clip(v, fixed_vmin, fixed_vmax)
            color = cmap(norm(bounded_v))
            hex_colors.append(rgb2hex(color))
    return np.array(hex_colors)

def definecolors_fixed_0_10(cells, property_name='parent_group', cmap_name='hls'):
    """
    Assign colors to cells based on a DISCRETE specified property,
    using a fixed mapping for values 0–10 so colors are consistent
    across calls.
    """
    if property_name not in cells.properties:
        cells.properties[property_name] = np.zeros(len(cells), dtype=int)

    prop_values = cells.properties[property_name]

    # Fixed range of groups (0–10)
    fixed_groups = np.arange(0, 11)  # 0,1,2,...,10

    cmap_size = len(fixed_groups)  # Always 11
    if cmap_name.lower() in ['hls', 'husl', 'deep', 'muted', 'pastel', 'bright', 'dark']:
        palette = sns.color_palette(cmap_name, cmap_size)
        cmap = ListedColormap([rgb2hex(color) for color in palette])
    else:
        cmap = cm.get_cmap(cmap_name, cmap_size)

    # Map each fixed group to its color
    color_dict = {group: rgb2hex(cmap(i)) for i, group in enumerate(fixed_groups)}

    # Assign colors to each cell (default to background color if out of range)
    colors = np.array([color_dict.get(val, color_dict[0]) for val in prop_values])

    return colors

def definecolors_manual(cells, property_name='parent_group', color_dict=None):
    """
    Asigna colores a cada valor de property_name según un diccionario manual.
    Si el valor no está en el diccionario, usa un color por defecto.
    """
    if property_name not in cells.properties:
        cells.properties[property_name] = np.zeros(len(cells), dtype=int)
    prop_values = cells.properties[property_name]
    # Diccionario de ejemplo: puedes poner los colores hex que quieras
    if color_dict is None:
        color_dict = {
            0: "#A1C9F4",  # normal cells
            1: "#F7A469FF",  # mutant cells
            2: "#A1C9F4",  # azul
            3: "#e0fb98",  # verde
            4: "#71b6b0",  # morado
            5: "#f0f0ee",  # naranja
            6: "#bab49e",  # amarillo
            7: "#b26801",  # marrón
            8: "#008080",  # rosa
            9: "#d4af37",  # gris
            10: "#ff7f50"  # cyan
        }
    # Asigna color según el diccionario, si no existe usa blanco
    colors = np.array([color_dict.get(val, "#ffffff") for val in prop_values])
    return colors

    
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm, rgb2hex

import numpy as np
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm, rgb2hex

def definecolors_fixed_1_10_center6(cells,
                                    property_name='value',
                                    cmap_name='RdBu_r'):
    """
    Assign DISCRETE colors to integer values 1–10,
    using a diverging colormap centered at 6.
    Mapping is fixed and consistent across calls.
    """

    if property_name not in cells.properties:
        cells.properties[property_name] = np.ones(len(cells), dtype=int) * 6

    prop_values = np.asarray(cells.properties[property_name], dtype=int)

    # Fixed groups 1–10
    fixed_groups = np.arange(1, 10)

    # Center at 6
    norm = TwoSlopeNorm(vmin=3, vcenter=6, vmax=9)
    cmap = cm.get_cmap(cmap_name)

    # Build fixed mapping dictionary
    color_dict = {
        group: rgb2hex(cmap(norm(group)))
        for group in fixed_groups
    }

    # Assign colors (fallback to 6 if out of range)
    colors = np.array([
        color_dict.get(val, color_dict[6])
        for val in prop_values
    ])

    return colors