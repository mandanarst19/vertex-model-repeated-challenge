# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from scipy.spatial import Voronoi

from vertex_lite.permutations import cycles, inverse
from vertex_lite.mesh import Edges, Mesh, Torus, cross2d


def swap_if(condition, a, b):
    for i, cond in enumerate(condition):
        if cond:
            a[i], b[i] = b[i], a[i]


def _correct_orientation(vor):
    face_centres = vor.points[vor.ridge_points]
    vertex_id_pairs = vor.ridge_vertices
    vertex_positions = vor.vertices[vertex_id_pairs]
    d = lambda pair: pair[:, 1] - pair[:, 0]
    swap_if(cross2d(d(face_centres), d(vertex_positions)) < 0,
            vertex_id_pairs[:, 0], vertex_id_pairs[:, 1])


def attach(data, from_cols, to_cols):
    ix = data[from_cols].argsort().argsort()
    return data[to_cols].argsort()[ix]


def voronoi(centres):
    vor = Voronoi(centres)
    vor.ridge_vertices = np.array(list(zip(*vor.ridge_vertices))).T  # simple np.array call is too slow..
    # edges are not consistently oriented so we fix this
    _correct_orientation(vor)
    return vor


def _edge_table(face_id_pairs, vertex_id_pairs, region_id_pairs):
    if region_id_pairs is None:
        region_id_pairs = np.zeros_like(face_id_pairs)

    edge_data = [face_id_pairs, vertex_id_pairs, region_id_pairs]
    # add reverse orientation of each edge
    flip = lambda pair: np.roll(pair, 1, axis=-1)
    edges = np.vstack([np.hstack(edge_data), np.hstack(list(map(flip, edge_data)))])
    dtype = {'names': ['face', 'face2', 'vertex', 'vertex2', 'region', 'region2'], 'formats': ['i8']*6}
    return edges.view(dtype).view(np.recarray)


def build_mesh(vertex_positions, geometry, face_id_pairs, vertex_id_pairs, region_id_pairs=None, boundary_face_ids=None):
    edge_data = _edge_table(face_id_pairs, vertex_id_pairs, region_id_pairs)
    fundamental_region = edge_data.region == 0
    edge_data = edge_data[fundamental_region]

    # build 'half-edge' representation
    nxt = attach(edge_data, ['face', 'vertex2'], ['face', 'vertex'])  # 'next' edge
    edge_data.region = -edge_data.region2
    reverse = attach(edge_data, ['face', 'face2', 'region'], ['face2', 'face', 'region2'])  # 'reverse' edge

    order = cycles(nxt[reverse])[0]  # order so that edges around a vertex are consecutive
    reverse = inverse(order)[reverse[order]]  # group conjugation of reverse by order
    vertices = vertex_positions[edge_data.vertex[order]].T.copy()

    edges = Edges(reverse)
    face_id_by_edge = cycles(edges.next)[1]

    boundary = None
    if boundary_face_ids is not None:
        face = edge_data.face[order]
        boundary_edges = np.any([face == face_id for face_id in boundary_face_ids], axis=0)
        boundary = np.unique(face_id_by_edge[boundary_edges])

    return Mesh(edges, vertices, face_id_by_edge, geometry, boundary_faces=boundary)


def toroidal_voronoi_mesh(centres, width, height):
    """Returns a Mesh data structure on a torus constructed as a voronoi diagram with the given centres.

    Args:
        centres: an (N,2) float array of x,y positions in the interval [-width/2,width/2]*[-height/2,height/2]
        width: a float giving periodicity in the x-direction
        height: a float giving periodicity in the y-direction
        Returns:
        A Mesh data structure.
    """
    centres_3x3 = np.vstack([centres+[dx, dy] for dx in [-width, 0, width] for dy in [-height, 0, height]])
    vor = voronoi(centres_3x3)

    N_cell = len(vor.points)//9
    region_id_pairs = vor.ridge_points//N_cell-4  # fundamental region is 0
    face_id_pairs = vor.ridge_points % N_cell  # idx mapped to fundamental region

    return build_mesh(vor.vertices, Torus(width, height), face_id_pairs, vor.ridge_vertices, region_id_pairs)


def random_centres(N_cell_across, N_cell_up, rand):
    N_cell, width, height = N_cell_across * N_cell_up, N_cell_across, N_cell_up
    a = rand.rand(N_cell, 2)-np.array([0.5, 0.5])  # uniform [-0.5,0.5]*[-0.5,0.5]
    b = (a*np.sqrt(N_cell/25)).astype(int)  # location on a coarse grid
    centres = a[np.lexsort((a[:, 0], b[:, 1], b[:, 0]))]  # sort by grid ref to improve locality
    centres = centres*np.array([width, height])
    return centres, width, height

def hexagonal_centres(N_cell_across, N_cell_up, noise, rand):
    assert(N_cell_up % 2 == 0)  # expect even number of rows
    dx, dy = 1.0/N_cell_across, 1.0/(N_cell_up/2)
    x = np.arange(-0.5+dx/4, 0.5, dx)
    y = np.arange(-0.5+dy/4, 0.5, dy)
    centres = np.zeros((N_cell_across, N_cell_up//2, 2, 2))
    centres[:, :, 0, 0] += x[:, np.newaxis]
    centres[:, :, 0, 1] += y[np.newaxis, :]
    x += dx/2
    y += dy/2
    centres[:, :, 1, 0] += x[:, np.newaxis]
    centres[:, :, 1, 1] += y[np.newaxis, :]

    ratio = np.sqrt(2/np.sqrt(3))
    width = N_cell_across*ratio
    height = N_cell_up/ratio

    centres = centres.reshape(-1, 2)*np.array([width, height])
    centres += rand.rand(N_cell_up*N_cell_across, 2)*noise
    return centres, width, height


def hexagonal_centres2(N_cell_across, N_cell_up, noise, rand):
    assert(N_cell_up % 2 == 0)  # expect even number of rows
    dx, dy = 1/N_cell_across, 1/(N_cell_up/2)
    x = np.arange(-0.5+dx/4, 0.5, dx)
    y = np.arange(-0.5+dy/4, 0.5, dy)
    centres = np.zeros((N_cell_across, N_cell_up//2, 2, 2))
    centres[:, :, 0, 0] += x[:, np.newaxis]
    centres[:, :, 0, 1] += y[np.newaxis, :]
    x += dx/2
    y += dy/2
    centres[:, :, 1, 0] += x[:, np.newaxis]
    centres[:, :, 1, 1] += y[np.newaxis, :]

    ratio = np.sqrt(2/np.sqrt(3))
    width = N_cell_across*ratio
    height = N_cell_up/ratio

    centres = centres.reshape(-1, 2)*np.array([width, height])

    distortion = rand.multivariate_normal([0,0], np.identity(2), size=[1,N_cell_up*N_cell_across])
    distortion=distortion[0]*noise
    
    # Adding noise to the centres
    #centres += rand.rand(N_cell_up*N_cell_across, 2)*noise
    # note that the parameter noise is the beta parameter used to control the distortion level
    # of the network in the sci report paper they use it from beta=0 to beta=100
   
    centres += distortion

    return centres, width, height

def toroidal_random_mesh(N_cell_across, N_cell_up, rand):
    return toroidal_voronoi_mesh(*random_centres(N_cell_across, N_cell_up, rand))


def toroidal_hex_mesh(N_cell_across, N_cell_up, noise=None, rand=None):
    return toroidal_voronoi_mesh(*hexagonal_centres2(N_cell_across, N_cell_up, noise, rand))


def toroidal_ic_mesh(ic):
    return toroidal_voronoi_mesh(*ic)

