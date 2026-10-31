## IMPORTS
from glob import glob
import os
import numpy as np
from scipy.spatial import Delaunay
from vertex_lite import Cells

def selected_neighbours(cells:Cells, cell_ids:list[int]) -> list[int]:
    """
    For a given list of cell_ids, return the set of all unique neighbours
    (excluding the original cell ids) by checking the boundary edges of each cell.
    """
    mesh = cells.mesh
    all_neighbours = set()
    for cell_id in cell_ids:
        if mesh.area[cell_id] <= 0:
         continue
        # Get the boundary edges for the current cell
        edges = mesh.boundary(cell_id)
        for e in edges:
            # Look up the reverse edge and the corresponding cell
            rev_e = mesh.edges.reverse[e]
            neighbor = mesh.face_id_by_edge[rev_e]
            
            # Exclude the current cell and invalid boundaries (-1)
            if neighbor != cell_id and neighbor != -1:
                all_neighbours.add(int(neighbor))
    # Exclude the original cell ids from the result:
    all_neighbours.difference_update(cell_ids)
    return list(all_neighbours)

def shape_index_norm(mesh, face_id=None,normalize=True):
    """
    Returns the mean of all cells' shape index, or the shape index for a single face if face_id is provided.
    """
    if face_id is not None:
        # <added> Wrap a non-iterable face_id into a list.
        if not hasattr(face_id, '__iter__'):
            face_id = [face_id]
        # Calculate shape index for the provided face(s)
        shape_index = []
        for i in face_id:
            l = mesh.perimeter[i]
            a = mesh.area[i]
            if a > 0 and l > 0:
                if normalize:
                    shape_index.append(l / (2*np.sqrt(2*a*np.sqrt(3))))
                else:
                    shape_index.append(l / (np.sqrt(a)))
        if shape_index:
            return np.mean(shape_index)
        else:
            return None  # Return None if area or perimeter is zero
    else:
        # Calculate mean shape index for all faces
        shape_index = []
        for j in range(len(mesh.area)):
            l = mesh.perimeter[j]
            a = mesh.area[j]
            if a > 0 and l > 0:
                if normalize:
                    shape_index.append(l / (2*np.sqrt(2*a*np.sqrt(3))))
                else:
                    shape_index.append(l / (np.sqrt(a)))
        return np.mean(shape_index)



def shape_index(cells:Cells, face_id=None,normalize=False, max_shape_index=5.0 ):
    """
    Returns the shape index for a single face if face_id is provided, or the mean shape index for all cells.
    
    Args:
        cells: Cells object containing mesh data.
        face_id: Optional; specific face ID to calculate shape index for.
        normalize: If True, normalizes the shape index.
        max_shape_index: Maximum value for shape index (default is 5).
    
    Returns:
        float: Shape index value(s).
    
    """
    if face_id is not None:
        if not hasattr(face_id, '__iter__'):
            face_id = [face_id]
        indices = face_id
    else:
        indices = range(len(cells.mesh.area))
    parent_group = cells.properties.get('parent_group', np.zeros(len(cells), dtype=int))
    shape_index = []
    for i in indices:
        l = cells.mesh.perimeter[i]
        a = cells.mesh.area[i]
        if a > 0 and l > 0:
            si = l / (2*np.sqrt(2*a*np.sqrt(3))) if normalize else l / np.sqrt(a)
            if si<= max_shape_index:
                shape_index.append(si)
            else:
                is_mutant=parent_group[i] == 1
                if not is_mutant:
                    print(f" WARNING: Non-mutant cell ID {i} has extreme shape index {si:.2f}")
    if shape_index:
        return np.mean(shape_index)
    else:
        return None          
    

    
def neighbour_groups(cells:Cells,cell_id:list[int],n_groups:int | None =None):
    cells.properties['neigh_groups'] = np.zeros(len(cells),dtype=int)
    cells.properties['neigh_groups'][cell_id] = 1
    current_group=1
    
    while np.any(cells.properties['neigh_groups'] == 0) and (n_groups is None or current_group < n_groups):
        

        
        selected_cells = np.where(cells.properties['neigh_groups'] == current_group)[0]
        
        new= selected_neighbours(cells, selected_cells)
        new= [nb for nb in new if cells.properties['neigh_groups'][nb] == 0]
        if not new:
            break
        current_group += 1
        cells.properties['neigh_groups'][new] = current_group
    return cells.properties['neigh_groups']



def multiple_neighbour_groups(cells:Cells, cell_ids:list[int], n_groups:int | None =None):
    
    """
    Calcula grupos de vecinos para múltiples células iniciales (mutantes).
    No modifica cells.properties, devuelve un diccionario con claves = IDs de células mutantes
    y valores = array con el número de grupo al que pertenece cada célula.

    Args:
        cells: objeto Cells
        cell_ids: lista de IDs de células mutantes iniciales (por ejemplo [1, 2])
        n_groups: número máximo de grupos a formar (opcional)

    Returns:
        dict: {cell_id1: groups1, cell_id2: groups2, ...}
    """
    if isinstance(cell_ids, (int, np.integer)):
        cell_ids = [int(cell_ids)]  # convierte a lista si es un número
    results = {}
    for cell_id in cell_ids:
        neigh_groups= np.zeros(len(cells), dtype=int)
        neigh_groups[cell_id] = 1
        current_group = 1   
        while np.any(neigh_groups == 0) and (n_groups is None or current_group < n_groups):
            selected_cells = np.where(neigh_groups == current_group)[0]
            new = selected_neighbours(cells, selected_cells)
            new = [nb for nb in new if neigh_groups[nb] == 0]
            if not new:
                break
            current_group += 1
            neigh_groups[new] = current_group
        results[cell_id] = neigh_groups
    return results



def cluster_mutants(cells:Cells, pct:float, rand:np.random.Generator,seed=None):
    """
    Select a clustered set of mutant cells.

    Strategy:
    - Pick one random seed cell.
    - Add its neighbours as mutants until no space is left.
    - If exhausted, compute neighbours of the whole mutant set and repeat.
    """
    N_total = len(cells)
    
    N_mutants = int(N_total * pct)
   
    if N_mutants == 0:
        return np.array([], dtype=int)

    # --- 1. Seed mutante inicial ---
    if seed is None:
        seed = rand.randint(N_total)
    else:
        seed = int(seed)  # Asegura que el seed sea un entero
    mutant_ids = set([seed])

    # Frontier = células desde las que crecer
   
    current_shell= selected_neighbours(cells,[seed])
    rand.shuffle(current_shell)
    # --- 2. Crecimiento del cluster ---
    while len(mutant_ids) < N_mutants :
       
        if not current_shell:
            current_shell = selected_neighbours(cells, list(mutant_ids))
            rand.shuffle(current_shell)
            
            if not current_shell:
                break  # No more space to grow  

        nb = current_shell.pop()
        if nb not in mutant_ids:
            mutant_ids.add(nb)
    return np.array(list(mutant_ids), dtype=int)



def calculate_interface_perimeter(cells:Cells):
    """
    Calculate the total perimeter of the interface between wildtype and mutant cells.
    
    Uses the half-edge mesh structure to efficiently identify and sum interface edges.
    
    Parameters:
    -----------
    cells : Cells object
        Must have:
        - mesh with edges, face_id_by_edge, edges.reverse, length
        - properties['parent_group'] with values 0 (WT) or 1 (mutant)
    
    Returns:
    --------
    interface_perimeter : float
        Total length of edges separating WT and mutant cells
    """
    mesh = cells.mesh
    parent_group = cells.properties['parent_group']
    
    interface_perimeter = 0.0
    
    # Iterate through all edges
    # We only need to check each undirected edge once
    # Convention: only count edges where edge_id < reverse_id
    for edge_id in range(len(mesh.edges)):
        reverse_edge = mesh.edges.reverse[edge_id]
        
        # Only process each undirected edge once
        if edge_id >= reverse_edge:
            continue
        
        # Get the two cells separated by this edge
        cell_a = mesh.face_id_by_edge[edge_id]
        cell_b = mesh.face_id_by_edge[reverse_edge]

        # Skip invalid cells
        if cell_a == -1 or cell_b == -1 or cell_a >= len(parent_group) or cell_b >= len(parent_group):
            continue
        
        # Check if this edge separates WT from mutant
        is_a_mutant = parent_group[cell_a] == 1
        is_b_mutant = parent_group[cell_b] == 1
        
        # Interface edge: one cell is mutant, the other is WT
        if is_a_mutant != is_b_mutant:
            interface_perimeter += mesh.length[edge_id]
    
    return interface_perimeter
 
 
def calculate_interface_perimeter_by_mutant(cells:Cells):
    """
    Calculate the perimeter of the interface around EACH mutant cell.
    
    Returns a dictionary mapping mutant_id -> interface_perimeter_touching_that_mutant
    
    Parameters:
    -----------
    cells : Cells object
    
    Returns:
    --------
    interface_dict : dict
        Keys: mutant cell IDs
        Values: total interface perimeter for that mutant (edges between that mutant and WT)
    """
    mesh = cells.mesh
    parent_group = cells.properties['parent_group']
    mutant_cells = np.where(parent_group == 1)[0]
    
    interface_dict = {m_id: 0.0 for m_id in mutant_cells}
    
    # Iterate through all edges
    for edge_id in range(len(mesh.edges)):
        reverse_edge = mesh.edges.reverse[edge_id]
        
        # Only process each undirected edge once
        if edge_id >= reverse_edge:
            continue
        
        # Get the two cells separated by this edge
        cell_a = mesh.face_id_by_edge[edge_id]
        cell_b = mesh.face_id_by_edge[reverse_edge]
        
        # Skip invalid cells
        if cell_a == -1 or cell_b == -1 or cell_a >= len(parent_group) or cell_b >= len(parent_group):
            continue
        
        # Check cell types
        is_a_mutant = parent_group[cell_a] == 1
        is_b_mutant = parent_group[cell_b] == 1
        
        # If interface edge (one mutant, one WT)
        if is_a_mutant and not is_b_mutant:
            interface_dict[cell_a] += mesh.length[edge_id]
        elif is_b_mutant and not is_a_mutant:
            interface_dict[cell_b] += mesh.length[edge_id]
    
    return interface_dict
 
 
def get_interface_edges(cells:Cells):
    """
    Get the list of all edges that form the WT-mutant interface.
    
    Useful for visualization or detailed analysis.
    
    Returns:
    --------
    interface_edges : list of int
        Edge IDs that separate WT from mutant cells
    interface_lengths : array
        Corresponding edge lengths
    """
    mesh = cells.mesh
    parent_group = cells.properties['parent_group']
    
    interface_edges = []
    interface_lengths = []
    
    for edge_id in range(len(mesh.edges)):
        reverse_edge = mesh.edges.reverse[edge_id]
        
        # Process each undirected edge once
        if edge_id >= reverse_edge:
            continue
        
        cell_a = mesh.face_id_by_edge[edge_id]
        cell_b = mesh.face_id_by_edge[reverse_edge]
        
        # Skip invalid cells
        if cell_a == -1 or cell_b == -1 or cell_a >= len(parent_group) or cell_b >= len(parent_group):
            continue
        
        is_a_mutant = parent_group[cell_a] == 1
        is_b_mutant = parent_group[cell_b] == 1
        
        if is_a_mutant != is_b_mutant:
            interface_edges.append(edge_id)
            interface_lengths.append(mesh.length[edge_id])
    
    return interface_edges, np.array(interface_lengths)

def calculate_t_eff(cells:Cells, cell_id:int | None=None):
    if cell_id is None:
        return None
    if not isinstance(cell_id, (int, np.integer)):
        raise ValueError("calculate_t_eff only supports a single cell_id at a time.")
    neigh = selected_neighbours(cells, [cell_id])
    if neigh:
        if not isinstance(cells.properties['Gamma'], np.ndarray):
            cells.properties['Gamma'] = np.array(len(cells)*[cells.properties['Gamma']])
        lambda_neigh = 0.5 * np.mean(cells.properties['Lambda'][neigh])
        P_times_G = np.mean(cells.mesh.perimeter[neigh] * cells.properties['Gamma'][neigh])
        t_eff = lambda_neigh + P_times_G + 0.5 * cells.properties['Lambda'][cell_id] + cells.mesh.perimeter[cell_id] * cells.properties['Gamma'][cell_id]
        return t_eff
    else:
        return np.nan


def calculate_center_mean(cells:Cells, cell_id:int):
    """
    Calculate the centroid of a given cell based on its boundary vertices.
    
    Parameters:
    -----------
    cells : Cells object
        Must have a mesh with vertices and boundary information.
    cell_id : int
        ID of the cell for which to calculate the centroid.
    
    Returns:
    --------
    centroid : tuple (x, y)
        Coordinates of the centroid. Returns (np.nan, np.nan) if the cell is invalid or has no area.
    """
    if cells.mesh.area[cell_id] <= 0:
        return (np.nan, np.nan)
    
    boundary_vertices = cells.mesh.boundary(cell_id)
    
    # Check for valid boundary vertices
    if len(boundary_vertices) < 3 or np.any(boundary_vertices == -1):
        return (np.nan, np.nan)
    
    vertex_coords = cells.mesh.vertices[:, boundary_vertices]
    
    # Calculate centroid as the mean of vertex coordinates
    centroid_x = np.mean(vertex_coords[0])
    centroid_y = np.mean(vertex_coords[1])
    
    return (centroid_x, centroid_y)

def calculate_centroid(cells:Cells, cell_id:int):
    """
    Calculate centroid of a cell using the shoelace formula.
    
    Parameters
    ----------
    cells : Cells
        Cell object with a mesh
    cell_id : int
        Cell ID to calculate centroid for
    
    Returns
    -------
    tuple
        (x_centroid, y_centroid)
    """
    mesh = cells.mesh
    if mesh.area[cell_id] <= 0:
        return (np.nan, np.nan)
    bound_edges = mesh.boundary(cell_id)
    
    if len(bound_edges) < 3:
        return (np.nan, np.nan)
    
    Area = mesh.area[cell_id]
    
    if Area == 0:
        return (np.nan, np.nan)
    
    x_sum = 0.0
    y_sum = 0.0
    
    # Sum over all edges in the cell boundary
    for edge in bound_edges:
        x_i = mesh.vertices[0, edge]
        y_i = mesh.vertices[1, edge]
        
        next_edge = mesh.edges.next[edge]
        x_next = mesh.vertices[0, next_edge]
        y_next = mesh.vertices[1, next_edge]
        
        # Cross product term: x_j * y_{j+1} - x_{j+1} * y_j
        cross = x_i * y_next - x_next * y_i
        
        # Accumulate centroid components
        x_sum += (x_i + x_next) * cross
        y_sum += (y_i + y_next) * cross
    
    # Apply the shoelace centroid formula: 1/(6*A) * sum
    x_centroid = x_sum / (6.0 * Area)
    y_centroid = y_sum / (6.0 * Area)
    
    return (x_centroid, y_centroid)


def load_data(file_pattern, columns=None, time_filter=None):
    import glob
    import pandas as pd
    dfs = []
    
    for file in glob.glob(file_pattern):
        try:
            df = pd.read_parquet(file, columns=columns)
            if df.empty:
                df = pd.DataFrame(0, index=range(time_filter), columns=columns)
                continue
            if time_filter is not None:
                df = df[df['time'] == time_filter]
                if df.empty:
                    continue
            parts= file.split(os.sep)
            df['mesh_type']= float(parts[-4].replace('noise_',''))
            df['pct_mutant']= float(parts[-3].replace('pct_',''))
            df['sim_id']= int(parts[-2].replace('sim_',''))

            dfs.append(df)
        except Exception:
        # cualquier parquet raro / vacío / corrupto → lo ignoras
            continue    
    return pd.concat(dfs, ignore_index=True) if  dfs else pd.DataFrame()
def load_data_v2(file_pattern, columns=None, time_filter=None):
    import glob
    import pandas as pd
    dfs = []
    for file in glob.glob(file_pattern):
        try:
            df = pd.read_parquet(file, columns=columns)
            
            if df.empty:
                
                print(f"File {file} is empty → creating fake row")
                row = {col: 0 for col in columns}
                if time_filter is not None and 'time' in columns:
                    row['time'] = time_filter
                df = pd.DataFrame([row], columns=columns)

            if time_filter is not None:
                df = df[df['time'] == time_filter]
                if df.empty:
                    continue

            parts = file.split(os.sep)
            df['mesh_type'] = float(parts[-4].replace('noise_', ''))
            df['pct_mutant'] = float(parts[-3].replace('pct_', ''))
            df['sim_id'] = int(parts[-2].replace('sim_', ''))

            dfs.append(df)

        except Exception as e:
            print(f"Error in {file}: {e}")
    return pd.concat(dfs, ignore_index=True) if  dfs else pd.DataFrame()

def load_data_v3(file_pattern, columns=None, time_filter=None):
    import glob
    import pandas as pd
    import os

    dfs = []

    for file in glob.glob(file_pattern):
        try:
            df = pd.read_parquet(file)

            #  If file has no columns treat as empty
            if df.shape[1] == 0:
                df = pd.DataFrame()

            if df.empty:
                if columns is None:
                    raise ValueError("columns must be provided")

                row = {col: 0 for col in columns}

                # FORCE correct time
                if time_filter is not None:
                    row['time'] = time_filter

                df = pd.DataFrame([row])

            else:
                # Only select columns AFTER confirming they exist
                if columns is not None:
                    df = df[columns]

            if time_filter is not None:
                if 'time' not in df.columns:
                    continue

                df = df[df['time'] == time_filter]

                if df.empty:
                    continue

            parts = file.split(os.sep)
            df['mesh_type'] = float(parts[-4].replace('noise_', ''))
            df['pct_mutant'] = float(parts[-3].replace('pct_', ''))
            df['sim_id'] = int(parts[-2].replace('sim_', ''))

            dfs.append(df)

        except Exception as e:
            print(f"Skipping {file}: {e}")  # 🔥 DON'T hide errors
            continue

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()