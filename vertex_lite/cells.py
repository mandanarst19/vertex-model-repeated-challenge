# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import vertex_lite.forces as F
from vertex_lite.mesh import Mesh

class Cells(object):
    def __init__(self, mesh:Mesh, properties=None):
        self.mesh:Mesh= mesh
        self.properties = properties or {}

    def copy(self):
        # mesh is 'immutable' (so we can cache shared computations) => no need to copy it
        return Cells(self.mesh, self.properties.copy())

    def __len__(self):
        return self.mesh.n_face

    def empty(self):
        # self.mesh._edge_lookup[self.mesh.area<0]=-1 ###########anyadido por mi!!!!!!!!! 
        return self.mesh._edge_lookup == -1

    def by_face(self, property_name, boundary_property_name=None):
        value = self.properties[property_name]
        if self.mesh.has_boundary():
            value = make_array(value, len(self))
            boundary_value = self.properties[boundary_property_name] if boundary_property_name else 0.0
            value[self.mesh.boundary_faces] = boundary_value
        return value

    def by_edge(self, property_name, boundary_property_name=None):
        value_by_face = self.by_face(property_name, boundary_property_name)
        if not hasattr(value_by_face, 'shape'):  # scalar
            return value_by_face
        return value_by_face.take(self.mesh.face_id_by_edge)
    
    def forces(self):
        combined_force = (F.TargetArea() + F.Tension() + F.Perimeter() + F.Pressure()).force(self)
        self.forces_data = combined_force
        return self.forces_data

    #The computed force is stored as an attribute of the Cells instance (i.e. self.forces_data). 
    # This makes it accessible as long as the instance exists.
    #You call this method to obtain the current (computed) force data. 
    #The forces_data is essentially a “cached” value that is tied directly to the specific object.

    def update_forces(cells, F):
        cells.properties['force'] = F.copy()
        # Optionally update stored mechanical parameters if you need to track changes
        cells.properties['K'] = cells.properties.get('K', None)
        cells.properties['A0'] = cells.properties.get('A0', None)
    #This method takes an already computed force array (passed as F) and stores a copy of it in the properties dictionary under the key 'force'.
    # It also ensures that the mechanical parameters 'K' and 'A0' exist in the properties.
    #The computed force is stored as part of the Cells’ properties dictionary instead of a direct attribute.
    #  The properties dictionary is used for values that characterize the cells (such as mechanical parameters or other user-defined metadata).
    
def make_array(value, n):
    if hasattr(value, 'shape'):
        return value
    expanded = np.empty(n, dtype=type(value))
    expanded.fill(value)
    return expanded

