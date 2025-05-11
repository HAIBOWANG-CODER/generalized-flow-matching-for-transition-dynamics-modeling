import jax
import jax.numpy as jnp
from jax import grad, jit
from Bio.PDB import PDBParser
import mdtraj as md
import os
import numpy as np
import glob
import openmm.app as app
import openmm.unit as unit
import torch
from tqdm import tqdm

# File was ported from:
# https://github.com/VincentStimper/boltzmann-generators/blob/2b177fc155f533933489b8fce8d6483ebad250d3/boltzgen/internal.py
"""
# This is the z-matrix for alanine
z_matrix = [
    (0, [1, 4, 6]),
    (1, [4, 6, 8]),
    (2, [1, 4, 0]),
    (3, [1, 4, 0]),
    (4, [6, 8, 14]),
    (5, [4, 6, 8]),

    (7, [6, 8, 4]),

    (9, [8, 6, 4]),
    (10, [8, 6, 4]),
    (11, [10, 8, 6]),
    (12, [10, 8, 11]),
    (13, [10, 8, 11]),

    (15, [14, 8, 16]),
    (16, [14, 8, 6]),
    (17, [16, 14, 15]),
    (18, [16, 14, 8]),
    (19, [18, 16, 14]),
    (20, [18, 16, 19]),
    (21, [18, 16, 19])
]
cart_indices = [8, 6, 14]
"""

z_matrix = [
(5, [4, 0, 21]), (6, [4, 0, 21]), (8, [6, 4, 9]), (9, [6, 4, 0]), (10, [9, 6, 4]),
(11, [10, 9, 12]), (12, [10, 9, 6]), (13, [12, 10, 14]), (14, [12, 19, 10]), (15, [14, 12, 19]),
(16, [15, 14, 12]), (17, [9, 6, 10]), (18, [17, 9, 19]), (19, [17, 9, 10]), (20, [19, 17, 14]),
(22, [21, 4, 0]), (26, [25, 23, 42]), (27, [25, 23, 42]), (29, [27, 25, 30]), (30, [27, 25, 23]),
(31, [30, 27, 25]), (32, [31, 30, 33]), (33, [31, 30, 27]), (34, [33, 31, 35]), (35, [33, 40, 31]),
(36, [35, 33, 40]), (37, [36, 35, 33]), (38, [30, 27, 31]), (39, [38, 30, 40]), (40, [38, 30, 31]),
(41, [40, 38, 35]), (43, [42, 25, 23]), (47, [46, 44, 54]), (48, [46, 44, 54]), (50, [48, 46, 51]),
(51, [48, 46, 44]), (52, [51, 48, 46]), (53, [51, 48, 52]), (55, [54, 46, 44]), (57, [65, 62, 60]),
(59, [57, 65, 56]), (61, [60, 56, 68]), (62, [60, 56, 68]), (64, [62, 60, 65]), (65, [62, 60, 56]),
(67, [65, 62, 57]), (69, [68, 60, 56]), (73, [72, 70, 83]), (74, [72, 70, 83]), (76, [74, 72, 77]),
(77, [74, 72, 70]), (79, [74, 77, 80]), (80, [77, 74, 72]), (81, [80, 77, 74]), (82, [80, 77, 81]),
(84, [83, 72, 70]), (88, [87, 85, 97]), (89, [87, 85, 97]), (90, [89, 87, 91]), (91, [89, 87, 85]),
(92, [91, 89, 87]), (93, [89, 87, 91]), (94, [93, 89, 87]), (95, [93, 89, 94]), (96, [93, 89, 94]),
(98, [97, 87, 85]), (103, [101, 99, 104]), (105, [104, 101, 99]), (109, [108, 106, 118]), (110, [108, 106, 118]),
(111, [110, 108, 112]), (112, [110, 108, 106]), (113, [112, 110, 108]), (114, [110, 108, 112]),
(115, [114, 110, 108]), (116, [114, 110, 115]), (117, [114, 110, 115]), (119, [118, 108, 106]),
(123, [122, 120, 142]), (124, [122, 120, 142]), (126, [124, 122, 127]), (127, [124, 122, 120]),
(128, [127, 124, 122]), (129, [128, 127, 133]), (130, [128, 127, 124]), (131, [130, 128, 133]),
(132, [133, 127, 128]), (133, [127, 124, 128]), (134, [133, 132, 128]), (135, [134, 133, 136]),
(136, [140, 132, 133]), (137, [136, 133, 138]), (138, [132, 130, 128]), (139, [138, 133, 136]),
(140, [138, 132, 133]), (141, [140, 136, 133]), (143, [142, 122, 120]), (150, [149, 147, 144]),
(151, [149, 147, 144]), (153, [151, 149, 154]), (154, [151, 149, 147]), (155, [154, 151, 149]),
(156, [155, 154, 157]), (157, [155, 154, 151]), (158, [157, 155, 159]), (159, [157, 164, 155]),
(160, [159, 157, 164]), (161, [160, 159, 157]), (162, [154, 151, 155]), (163, [162, 154, 164]),
(164, [162, 154, 155]), (165, [164, 162, 159])
]

cart_indices = [1, 2, 3, 4, 5, 8, 22, 24, 25, 26, 29, 43, 45, 46, 47, 50, 55, 57, 59, 61, 64, 67, 69, 71, 72, 73,
            76, 79, 84, 86, 87, 88, 98, 100, 101, 102, 103, 105, 107, 108, 109, 119, 121, 122, 123, 126, 143,
            145, 146, 147, 148, 149, 150, 153]


def calc_bonds(ind1, ind2, coords):
    """Calculate bond lengths

    Parameters
    ----------
    ind1 : jnp.ndarray
        A n_bond x 3 array of indices for the coordinates of particle 1
    ind2 : jnp.ndarray
        A n_bond x 3 array of indices for the coordinates of particle 2
    coords : jnp.ndarray
        A n_batch x n_coord array of flattened input coordinates
    """
    p1 = coords[:, ind1]
    p2 = coords[:, ind2]
    return jnp.linalg.norm(p2 - p1, axis=2)


def calc_angles(ind1, ind2, ind3, coords):
    b = coords[:, ind1]
    c = coords[:, ind2]
    d = coords[:, ind3]
    bc = b - c
    bc /= jnp.linalg.norm(bc, axis=2, keepdims=True)
    cd = d - c
    cd /= jnp.linalg.norm(cd, axis=2, keepdims=True)
    cos_angle = jnp.sum(bc * cd, axis=2)
    angle = jnp.arccos(cos_angle)
    return angle


def calc_dihedrals(ind1, ind2, ind3, ind4, coords):
    a = coords[:, ind1]
    b = coords[:, ind2]
    c = coords[:, ind3]
    d = coords[:, ind4]

    b0 = a - b
    b1 = c - b
    b1 /= jnp.linalg.norm(b1, axis=2, keepdims=True)
    b2 = d - c

    v = b0 - jnp.sum(b0 * b1, axis=2, keepdims=True) * b1
    w = b2 - jnp.sum(b2 * b1, axis=2, keepdims=True) * b1
    x = jnp.sum(v * w, axis=2)
    b1xv = jnp.cross(b1, v, axis=2)
    y = jnp.sum(b1xv * w, axis=2)
    angle = jnp.arctan2(y, x)
    return -angle


def reconstruct_cart(cart, ref_atoms, bonds, angles, dihs):
    # Get the positions of the 4 reconstructing atoms
    p1 = cart[:, ref_atoms[:, 0], :]
    p2 = cart[:, ref_atoms[:, 1], :]
    p3 = cart[:, ref_atoms[:, 2], :]

    bonds = jnp.expand_dims(bonds, axis=2)
    angles = jnp.expand_dims(angles, axis=2)
    dihs = jnp.expand_dims(dihs, axis=2)

    # Reconstruct the position of p4
    v1 = p1 - p2
    v2 = p1 - p3

    n = jnp.cross(v1, v2, axis=2)
    n = n / jnp.linalg.norm(n, axis=2, keepdims=True)
    nn = jnp.cross(v1, n, axis=2)
    nn = nn / jnp.linalg.norm(nn, axis=2, keepdims=True)

    n = n * jnp.sin(dihs)
    nn = nn * jnp.cos(dihs)

    v3 = n + nn
    v3 = v3 / jnp.linalg.norm(v3, axis=2, keepdims=True)
    v3 = v3 * bonds * jnp.sin(angles)

    v1 = v1 / jnp.linalg.norm(v1, axis=2, keepdims=True)
    v1 = v1 * bonds * jnp.cos(angles)

    # Store the final position in x
    new_cart = p1 + v3 - v1

    return new_cart


class InternalCoordinateTransform:
    def __init__(self, dims, z_indices=None, cart_indices=None, data=None,
                 ind_circ_dih=[], shift_dih=False,
                 shift_dih_params={'hist_bins': 100},
                 default_std={'bond': 0.005, 'angle': 0.15, 'dih': 0.2}):
        self.dims = dims
        # Setup indexing.
        self._setup_indices(z_indices, cart_indices)
        self._validate_data(data)
        # Setup the mean and standard deviations for each internal coordinate.
        transformed = self._fwd(data)
        # Normalize
        self.default_std = default_std
        self.ind_circ_dih = ind_circ_dih
        self._setup_mean_bonds(transformed)
        transformed = transformed.at[:, self.bond_indices].set(transformed[:, self.bond_indices] - self.mean_bonds)
        self._setup_std_bonds(transformed)
        transformed = transformed.at[:, self.bond_indices].set(transformed[:, self.bond_indices] / self.std_bonds)
        self._setup_mean_angles(transformed)
        transformed = transformed.at[:, self.angle_indices].set(transformed[:, self.angle_indices] - self.mean_angles)
        self._setup_std_angles(transformed)
        transformed = transformed.at[:, self.angle_indices].set(transformed[:, self.angle_indices] / self.std_angles)
        self._setup_mean_dih(transformed)
        transformed = transformed.at[:, self.dih_indices].set(transformed[:, self.dih_indices] - self.mean_dih)
        transformed = self._fix_dih(transformed)
        self._setup_std_dih(transformed)
        transformed = transformed.at[:, self.dih_indices].set(transformed[:, self.dih_indices] / self.std_dih)
        if shift_dih:
            val = jnp.linspace(-jnp.pi, jnp.pi,
                               shift_dih_params['hist_bins'])
            for i in self.ind_circ_dih:
                dih = transformed[:, self.dih_indices[i]]
                dih = dih * self.std_dih[i] + self.mean_dih[i]
                dih = (dih + jnp.pi) % (2 * jnp.pi) - jnp.pi
                hist = jnp.histogram(dih, bins=shift_dih_params['hist_bins'],
                                     range=(-jnp.pi, jnp.pi))[0]
                self.mean_dih = self.mean_dih.at[i].set(val[jnp.argmin(hist)] + jnp.pi)
                dih = (dih - self.mean_dih[i]) / self.std_dih[i]
                dih = (dih + jnp.pi) % (2 * jnp.pi) - jnp.pi
                transformed = transformed.at[:, self.dih_indices[i]].set(dih)

    def to_internal(self, x):
        trans = self._fwd(x)
        trans = trans.at[:, self.bond_indices].set(trans[:, self.bond_indices] - self.mean_bonds)
        trans = trans.at[:, self.bond_indices].set(trans[:, self.bond_indices] / self.std_bonds)
        trans = trans.at[:, self.angle_indices].set(trans[:, self.angle_indices] - self.mean_angles)
        trans = trans.at[:, self.angle_indices].set(trans[:, self.angle_indices] / self.std_angles)
        trans = trans.at[:, self.dih_indices].set(trans[:, self.dih_indices] - self.mean_dih)
        trans = self._fix_dih(trans)
        trans = trans.at[:, self.dih_indices].set(trans[:, self.dih_indices] / self.std_dih)
        return trans

    def _fwd(self, x):
        # we can do everything in parallel...
        inds1 = self.inds_for_atom[self.rev_z_indices[:, 1]]
        inds2 = self.inds_for_atom[self.rev_z_indices[:, 2]]
        inds3 = self.inds_for_atom[self.rev_z_indices[:, 3]]
        inds4 = self.inds_for_atom[self.rev_z_indices[:, 0]]

        # Calculate the bonds, angles, and torsions for a batch.
        bonds = calc_bonds(inds1, inds4, coords=x)
        angles = calc_angles(inds2, inds1, inds4, coords=x)
        dihedrals = calc_dihedrals(inds3, inds2, inds1, inds4, coords=x)

        # Replace the cartesian coordinates with internal coordinates.
        x = x.at[:, inds4[:, 0]].set(bonds)
        x = x.at[:, inds4[:, 1]].set(angles)
        x = x.at[:, inds4[:, 2]].set(dihedrals)
        return x

    def to_cartesian(self, x):
        # Gather all of the atoms represented as Cartesian coordinates.
        n_batch = x.shape[0]
        cart = x[:, self.init_cart_indices].reshape(n_batch, -1, 3)

        # Loop over all of the blocks, where all of the atoms in each block
        # can be built in parallel because they only depend on atoms that
        # are already Cartesian. `atoms_to_build` lists the `n` atoms
        # that can be built as a batch, where the indexing refers to the
        # original atom order. `ref_atoms` has size n x 3, where the indexing
        # refers to the position in `cart`, rather than the original order.
        for block in self.rev_blocks:
            atoms_to_build = block[:, 0]
            ref_atoms = block[:, 1:]

            # Get all of the bonds by retrieving the appropriate columns and
            # un-normalizing.
            bonds = (
                    x[:, 3 * atoms_to_build]
                    * self.std_bonds[self.atom_to_stats[atoms_to_build]]
                    + self.mean_bonds[self.atom_to_stats[atoms_to_build]]
            )

            # Get all of the angles by retrieving the appropriate columns and
            # un-normalizing.
            angles = (
                    x[:, 3 * atoms_to_build + 1]
                    * self.std_angles[self.atom_to_stats[atoms_to_build]]
                    + self.mean_angles[self.atom_to_stats[atoms_to_build]]
            )
            # Get all of the dihedrals by retrieving the appropriate columns and
            # un-normalizing.
            dihs = (
                    x[:, 3 * atoms_to_build + 2]
                    * self.std_dih[self.atom_to_stats[atoms_to_build]]
                    + self.mean_dih[self.atom_to_stats[atoms_to_build]]
            )

            # Fix the dihedrals to lie in [-pi, pi].
            dihs = jnp.where(dihs < jnp.pi, dihs + 2 * jnp.pi, dihs)
            dihs = jnp.where(dihs > jnp.pi, dihs - 2 * jnp.pi, dihs)

            # Compute the Cartesian coordinates for the newly placed atoms.
            new_cart = reconstruct_cart(cart, ref_atoms, bonds, angles, dihs)

            # Concatenate the Cartesian coordinates for the newly placed
            # atoms onto the full set of Cartesian coordinates.
            cart = jnp.concatenate([cart, new_cart], axis=1)
        # Permute cart back into the original order and flatten.
        cart = cart[:, self.rev_perm_inv]
        cart = cart.reshape(n_batch, -1)
        return cart

    def _setup_mean_bonds(self, x):
        self.mean_bonds = jnp.mean(x[:, self.bond_indices], axis=0)     # axis=1

    def _setup_std_bonds(self, x):
        if x.shape[0] > 1:
            self.std_bonds = jnp.std(x[:, self.bond_indices], axis=0)   # axis=0
        else:
            self.std_bonds = jnp.ones_like(self.mean_bonds) * self.default_std['bond']

    def _setup_mean_angles(self, x):
        self.mean_angles = jnp.mean(x[:, self.angle_indices], axis=0)

    def _setup_std_angles(self, x):
        if x.shape[0] > 1:
            self.std_angles = jnp.std(x[:, self.angle_indices], axis=0)
        else:
            self.std_angles = jnp.ones_like(self.mean_angles) * self.default_std['angle']

    def _setup_mean_dih(self, x):
        sin = jnp.mean(jnp.sin(x[:, self.dih_indices]), axis=0)
        cos = jnp.mean(jnp.cos(x[:, self.dih_indices]), axis=0)
        self.mean_dih = jnp.arctan2(sin, cos)

    def _fix_dih(self, x):

        dih = x[:, self.dih_indices]
        dih = (dih + jnp.pi) % (2 * jnp.pi) - jnp.pi
        x = x.at[:, self.dih_indices].set(dih)
        return x

    def _setup_std_dih(self, x):
        if x.shape[0] > 1:
            #self.std_dih = jnp.std(x.at[:, self.dih_indices], axis=0)          # ======================================================= #
            self.std_dih = jnp.std(x[:, self.dih_indices], axis=0)
        else:
            self.std_dih = jnp.ones_like(self.mean_dih) * self.default_std['dih']
            if len(self.ind_circ_dih) > 0:
                self.std_dih = self.std_dih.at[jnp.array(self.ind_circ_dih)].set(1.)

    def _validate_data(self, data):
        if data is None:
            raise ValueError(
                "InternalCoordinateTransform must be supplied with training_data."
            )

        if len(data.shape) != 2:
            raise ValueError("training_data must be n_samples x n_dim array")

        n_dim = data.shape[1]

        if n_dim != self.dims:
            raise ValueError(
                f"training_data must have {self.dims} dimensions, not {n_dim}."
            )

    def _setup_indices(self, z_indices, cart_indices):
        n_atoms = self.dims // 3
        ind_for_atom = jnp.zeros((n_atoms, 3), dtype=jnp.int32)
        for i in range(n_atoms):
            ind_for_atom = ind_for_atom.at[i].set([3 * i, 3 * i + 1, 3 * i + 2])
        self.inds_for_atom = ind_for_atom

        sorted_z_indices = topological_sort(z_indices)
        sorted_z_indices = [
            [item[0], item[1][0], item[1][1], item[1][2]] for item in sorted_z_indices
        ]
        rev_z_indices = list(reversed(sorted_z_indices))

        mod = [item[0] for item in sorted_z_indices]
        modified_indices = []
        for index in mod:
            modified_indices.extend(self.inds_for_atom[index])
        bond_indices = list(modified_indices[0::3])
        angle_indices = list(modified_indices[1::3])
        dih_indices = list(modified_indices[2::3])

        self.modified_indices = jnp.array(modified_indices)
        self.bond_indices = jnp.array(bond_indices)
        self.angle_indices = jnp.array(angle_indices)
        self.dih_indices = jnp.array(dih_indices)
        self.sorted_z_indices = jnp.array(sorted_z_indices)
        self.rev_z_indices = jnp.array(rev_z_indices)

        #
        # Setup indexing for reverse pass.
        #
        # First, create an array that maps from an atom index into mean_bonds, std_bonds, etc.
        atom_to_stats = jnp.zeros(n_atoms, dtype=jnp.int32)
        for i, j in enumerate(mod):
            atom_to_stats = atom_to_stats.at[j].set(i)
        self.atom_to_stats = atom_to_stats

        # Next create permutation vector that is used in the reverse pass. This maps
        # from the original atom indexing to the order that the Cartesian coordinates
        # will be built in. This will be filled in as we go.
        rev_perm = jnp.zeros(n_atoms, dtype=jnp.int32)
        self.rev_perm = rev_perm
        # Next create the inverse of rev_perm. This will be filled in as we go.
        rev_perm_inv = jnp.zeros(n_atoms, dtype=jnp.int32)
        self.rev_perm_inv = rev_perm_inv

        # Create the list of columns that form our initial Cartesian coordinates.
        init_cart_indices = self.inds_for_atom[jnp.array(cart_indices)].reshape(-1)
        self.init_cart_indices = init_cart_indices

        # Update our permutation vectors for the initial Cartesian atoms.
        for i, j in enumerate(cart_indices):
            self.rev_perm = self.rev_perm.at[i].set(j)
            self.rev_perm_inv = self.rev_perm_inv.at[j].set(i)

        # Break Z into blocks, where all of the atoms within a block
        # can be built in parallel, because they only depend on
        # atoms that are already Cartesian.
        all_cart = set(cart_indices)
        current_cart_ind = i + 1
        blocks = []
        while sorted_z_indices:
            next_z_indices = []
            next_cart = set()
            block = []
            for atom1, atom2, atom3, atom4 in sorted_z_indices:
                if (atom2 in all_cart) and (atom3 in all_cart) and (atom4 in all_cart):
                    # We can build this atom from existing Cartesian atoms,
                    # so we add it to the list of Cartesian atoms available for the next block.
                    next_cart.add(atom1)

                    # Add this atom to our permutation matrices.
                    self.rev_perm = self.rev_perm.at[current_cart_ind].set(atom1)
                    self.rev_perm_inv = self.rev_perm_inv.at[atom1].set(current_cart_ind)
                    current_cart_ind += 1

                    # Next, we convert the indices for atoms2-4 from their normal values
                    # to the appropriate indices to index into the Cartesian array.
                    atom2_mod = self.rev_perm_inv[atom2]
                    atom3_mod = self.rev_perm_inv[atom3]
                    atom4_mod = self.rev_perm_inv[atom4]

                    # Finally, we append this information to the current block.
                    block.append([atom1, atom2_mod, atom3_mod, atom4_mod])
                else:
                    # We can't build this atom from existing Cartesian atoms,
                    # so put it on the list for next time.
                     next_z_indices.append([atom1, atom2, atom3, atom4])
            sorted_z_indices = next_z_indices
            all_cart = all_cart.union(next_cart)
            block = jnp.array(block)
            blocks.append(block)
        self.rev_blocks = blocks


def topological_sort(graph_unsorted):
    graph_sorted = []
    graph_unsorted = dict(graph_unsorted)

    while graph_unsorted:
        acyclic = False
        for node, edges in list(graph_unsorted.items()):
            for edge in edges:
                if edge in graph_unsorted:
                    break
            else:
                acyclic = True
                del graph_unsorted[node]
                graph_sorted.append((node, edges))

        if not acyclic:
            raise RuntimeError("A cyclic dependency occured.")

    return graph_sorted


def read_pdb(paths):

    all_data = []
    pdb = app.PDBFile(paths)
    data_x = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    adata = data_x.flatten()
    if len(adata) != 498:
        adata = np.pad(adata, (0, 498 - len(adata)), 'constant')
    all_data.append(adata)
    combined_data = np.vstack(all_data)
    combined_data = jnp.array(combined_data)

    return combined_data


def internal_to_cartesian(inter_data):
    z_matrix = [
        (5, [4, 0, 21]), (6, [4, 0, 21]), (8, [6, 4, 9]), (9, [6, 4, 0]), (10, [9, 6, 4]),
        (11, [10, 9, 12]), (12, [10, 9, 6]), (13, [12, 10, 14]), (14, [12, 19, 10]), (15, [14, 12, 19]),
        (16, [15, 14, 12]), (17, [9, 6, 10]), (18, [17, 9, 19]), (19, [17, 9, 10]), (20, [19, 17, 14]),
        (22, [21, 4, 0]), (26, [25, 23, 42]), (27, [25, 23, 42]), (29, [27, 25, 30]), (30, [27, 25, 23]),
        (31, [30, 27, 25]), (32, [31, 30, 33]), (33, [31, 30, 27]), (34, [33, 31, 35]), (35, [33, 40, 31]),
        (36, [35, 33, 40]), (37, [36, 35, 33]), (38, [30, 27, 31]), (39, [38, 30, 40]), (40, [38, 30, 31]),
        (41, [40, 38, 35]), (43, [42, 25, 23]), (47, [46, 44, 54]), (48, [46, 44, 54]), (50, [48, 46, 51]),
        (51, [48, 46, 44]), (52, [51, 48, 46]), (53, [51, 48, 52]), (55, [54, 46, 44]), (57, [65, 62, 60]),
        (59, [57, 65, 56]), (61, [60, 56, 68]), (62, [60, 56, 68]), (64, [62, 60, 65]), (65, [62, 60, 56]),
        (67, [65, 62, 57]), (69, [68, 60, 56]), (73, [72, 70, 83]), (74, [72, 70, 83]), (76, [74, 72, 77]),
        (77, [74, 72, 70]), (79, [74, 77, 80]), (80, [77, 74, 72]), (81, [80, 77, 74]), (82, [80, 77, 81]),
        (84, [83, 72, 70]), (88, [87, 85, 97]), (89, [87, 85, 97]), (90, [89, 87, 91]), (91, [89, 87, 85]),
        (92, [91, 89, 87]), (93, [89, 87, 91]), (94, [93, 89, 87]), (95, [93, 89, 94]), (96, [93, 89, 94]),
        (98, [97, 87, 85]), (103, [101, 99, 104]), (105, [104, 101, 99]), (109, [108, 106, 118]),
        (110, [108, 106, 118]),
        (111, [110, 108, 112]), (112, [110, 108, 106]), (113, [112, 110, 108]), (114, [110, 108, 112]),
        (115, [114, 110, 108]), (116, [114, 110, 115]), (117, [114, 110, 115]), (119, [118, 108, 106]),
        (123, [122, 120, 142]), (124, [122, 120, 142]), (126, [124, 122, 127]), (127, [124, 122, 120]),
        (128, [127, 124, 122]), (129, [128, 127, 133]), (130, [128, 127, 124]), (131, [130, 128, 133]),
        (132, [133, 127, 128]), (133, [127, 124, 128]), (134, [133, 132, 128]), (135, [134, 133, 136]),
        (136, [140, 132, 133]), (137, [136, 133, 138]), (138, [132, 130, 128]), (139, [138, 133, 136]),
        (140, [138, 132, 133]), (141, [140, 136, 133]), (143, [142, 122, 120]), (150, [149, 147, 144]),
        (151, [149, 147, 144]), (153, [151, 149, 154]), (154, [151, 149, 147]), (155, [154, 151, 149]),
        (156, [155, 154, 157]), (157, [155, 154, 151]), (158, [157, 155, 159]), (159, [157, 164, 155]),
        (160, [159, 157, 164]), (161, [160, 159, 157]), (162, [154, 151, 155]), (163, [162, 154, 164]),
        (164, [162, 154, 155]), (165, [164, 162, 159])
    ]

    cart_indices = [1, 2, 3, 4, 5, 8, 22, 24, 25, 26, 29, 43, 45, 46, 47, 50, 55, 57, 59, 61, 64, 67, 69, 71, 72, 73,
                    76, 79, 84, 86, 87, 88, 98, 100, 101, 102, 103, 105, 107, 108, 109, 119, 121, 122, 123, 126, 143,
                    145, 146, 147, 148, 149, 150, 153]

    reference_pdb = r"C:\Users\Administrator\Documents\WeChat Files\wxid_iagpqtxayzg922\FileStorage\File\2025-04\test.pdb"  # Reference PDB data
    reference_cartesian_coords = read_pdb(reference_pdb)
    dims = 498
    transformer = InternalCoordinateTransform(
        dims=dims,
        z_indices=z_matrix,
        cart_indices=cart_indices,
        data=reference_cartesian_coords
    )

    if not torch.is_tensor(inter_data):
        inter_data = torch.tensor(inter_data)

    device = torch.device("cpu")
    inter_coords = inter_data.reshape(-1, dims).to(device)

    inter_coords = jnp.array(inter_coords)
    cart_coords_ex = transformer.to_cartesian(inter_coords)
    cart_np = np.array(cart_coords_ex)
    cart_tensor = torch.tensor(cart_np)

    if inter_data.dim() == 3:
        cart_tensor = cart_tensor.reshape(inter_data.shape[0], -1, dims)
    elif inter_data.dim() == 2:
        cart_tensor = cart_tensor.reshape(-1, dims)
    else:
        print("dim:", inter_data.dim())
        print('The dimension of the data is neither 3 nor 2')

    # torch.save(cartesian_tensor, save_path)
    # print(f"Cartesian coordinates saved to {save_path}")
    return cart_tensor

def cartesian_to_internal(cart_data):
    z_matrix = [
    (5, [4, 0, 21]), (6, [4, 0, 21]), (8, [6, 4, 9]), (9, [6, 4, 0]), (10, [9, 6, 4]),
    (11, [10, 9, 12]), (12, [10, 9, 6]), (13, [12, 10, 14]), (14, [12, 19, 10]), (15, [14, 12, 19]),
    (16, [15, 14, 12]), (17, [9, 6, 10]), (18, [17, 9, 19]), (19, [17, 9, 10]), (20, [19, 17, 14]),
    (22, [21, 4, 0]), (26, [25, 23, 42]), (27, [25, 23, 42]), (29, [27, 25, 30]), (30, [27, 25, 23]),
    (31, [30, 27, 25]), (32, [31, 30, 33]), (33, [31, 30, 27]), (34, [33, 31, 35]), (35, [33, 40, 31]),
    (36, [35, 33, 40]), (37, [36, 35, 33]), (38, [30, 27, 31]), (39, [38, 30, 40]), (40, [38, 30, 31]),
    (41, [40, 38, 35]), (43, [42, 25, 23]), (47, [46, 44, 54]), (48, [46, 44, 54]), (50, [48, 46, 51]),
    (51, [48, 46, 44]), (52, [51, 48, 46]), (53, [51, 48, 52]), (55, [54, 46, 44]), (57, [65, 62, 60]),
    (59, [57, 65, 56]), (61, [60, 56, 68]), (62, [60, 56, 68]), (64, [62, 60, 65]), (65, [62, 60, 56]),
    (67, [65, 62, 57]), (69, [68, 60, 56]), (73, [72, 70, 83]), (74, [72, 70, 83]), (76, [74, 72, 77]),
    (77, [74, 72, 70]), (79, [74, 77, 80]), (80, [77, 74, 72]), (81, [80, 77, 74]), (82, [80, 77, 81]),
    (84, [83, 72, 70]), (88, [87, 85, 97]), (89, [87, 85, 97]), (90, [89, 87, 91]), (91, [89, 87, 85]),
    (92, [91, 89, 87]), (93, [89, 87, 91]), (94, [93, 89, 87]), (95, [93, 89, 94]), (96, [93, 89, 94]),
    (98, [97, 87, 85]), (103, [101, 99, 104]), (105, [104, 101, 99]), (109, [108, 106, 118]), (110, [108, 106, 118]),
    (111, [110, 108, 112]), (112, [110, 108, 106]), (113, [112, 110, 108]), (114, [110, 108, 112]),
    (115, [114, 110, 108]), (116, [114, 110, 115]), (117, [114, 110, 115]), (119, [118, 108, 106]),
    (123, [122, 120, 142]), (124, [122, 120, 142]), (126, [124, 122, 127]), (127, [124, 122, 120]),
    (128, [127, 124, 122]), (129, [128, 127, 133]), (130, [128, 127, 124]), (131, [130, 128, 133]),
    (132, [133, 127, 128]), (133, [127, 124, 128]), (134, [133, 132, 128]), (135, [134, 133, 136]),
    (136, [140, 132, 133]), (137, [136, 133, 138]), (138, [132, 130, 128]), (139, [138, 133, 136]),
    (140, [138, 132, 133]), (141, [140, 136, 133]), (143, [142, 122, 120]), (150, [149, 147, 144]),
    (151, [149, 147, 144]), (153, [151, 149, 154]), (154, [151, 149, 147]), (155, [154, 151, 149]),
    (156, [155, 154, 157]), (157, [155, 154, 151]), (158, [157, 155, 159]), (159, [157, 164, 155]),
    (160, [159, 157, 164]), (161, [160, 159, 157]), (162, [154, 151, 155]), (163, [162, 154, 164]),
    (164, [162, 154, 155]), (165, [164, 162, 159])
]

    cart_indices = [1, 2, 3, 4, 5, 8, 22, 24, 25, 26, 29, 43, 45, 46, 47, 50, 55, 57, 59, 61, 64, 67, 69, 71, 72, 73,
                76, 79, 84, 86, 87, 88, 98, 100, 101, 102, 103, 105, 107, 108, 109, 119, 121, 122, 123, 126, 143,
                145, 146, 147, 148, 149, 150, 153]

    reference_pdb = r"C:\Users\Administrator\Documents\WeChat Files\wxid_iagpqtxayzg922\FileStorage\File\2025-04\test.pdb"   # Reference PDB data
    reference_cartesian_coords = read_pdb(reference_pdb)
    dims = 498
    transformer = InternalCoordinateTransform(
        dims=dims,
        z_indices=z_matrix,
        cart_indices=cart_indices,
        data=reference_cartesian_coords
    )

    if not torch.is_tensor(cart_data):
        cart_data = torch.tensor(cart_data)

    cart_coords = cart_data.reshape(-1, dims)

    cart_coords = jnp.array(cart_coords)
    cart_coords_ex = transformer.to_internal(cart_coords)
    cart_np = np.array(cart_coords_ex)
    cart_tensor = torch.tensor(cart_np)

    if cart_data.dim() == 3:
        cart_tensor = cart_tensor.reshape(cart_data.shape[0], -1, dims)
    elif cart_data.dim() == 2:
        cart_tensor = cart_tensor.reshape(-1, dims)
    else:
        print("dim:", cart_data.dim())
        print('The dimension of the data is neither 3 nor 2')

    # torch.save(cartesian_tensor, save_path)
    # print(f"Cartesian coordinates saved to {save_path}")
    return cart_tensor


def read_all_pdb_files_in_directory(paths):

    all_data = []
    for i, path in enumerate(paths):
        pdb_files = glob.glob(os.path.join(path, '*.pdb'))

        for pdb_file in pdb_files:

            pdb = app.PDBFile(pdb_file)
            data_x = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            adata = data_x.flatten()
            if len(adata) != 66:
                adata = np.pad(adata, (0, 66 - len(adata)), 'constant')

            all_data.append(adata)

    combined_data = np.vstack(all_data)
    combined_data = jnp.array(combined_data)

    return combined_data


if __name__ == "__main__":
    # ================================ initialize ===================================== #
    # reference_pdb = r"pdb_data\AD_A.pdb"                # Reference PDB data
    reference_pdb = r"C:\Users\Administrator\Documents\WeChat Files\wxid_iagpqtxayzg922\FileStorage\File\2025-04\test.pdb"
    reference_cartesian_coords = read_pdb(reference_pdb)
    dims = 498
    transformer = InternalCoordinateTransform(
        dims=dims,
        z_indices=z_matrix,
        cart_indices=cart_indices,
        data=reference_cartesian_coords
    )

    # ================================ cartesian to internal ===================================== #
    # pdb_file_path = [r"pdb_data/x0s"]
    # cartesian_coords = read_all_pdb_files_in_directory(pdb_file_path)      # Loading PDB data (Cartesian coordinate)
    folder_path = r"E:\chignolin_results\DEShaw_research_chignolin\DESRES-Trajectory_CLN025-0-protein\DESRES-Trajectory_CLN025-0-protein\CLN025-0-protein"
    topology_file = r"C:\Users\Administrator\Documents\WeChat Files\wxid_iagpqtxayzg922\FileStorage\File\2025-04\test.pdb"

    x0_indices = np.load(r"C:\Users\Administrator\Desktop\GFM\x0.npy")
    x1_indices = np.load(r"C:\Users\Administrator\Desktop\GFM\x1.npy")
    # pt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]
    pt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcd')]
    pt_files.sort()

    traj = md.load(pt_files, top=topology_file)
    x = traj.xyz

    x0_indices = np.random.choice(x0_indices, size=10000, replace=False)
    x1_indices = np.random.choice(x1_indices, size=10000, replace=False)
    x0_val_indices = np.random.choice(x0_indices, size=10000, replace=False)
    x1_val_indices = np.random.choice(x1_indices, size=10000, replace=False)

    cartesian_coords = x[x1_indices, :, :].reshape(-1, 498)
    cartesian_coords = jnp.array(cartesian_coords)
    
    internal_coords = transformer.to_internal(cartesian_coords)

    internal_np = np.array(internal_coords)
    internal_tensor = torch.tensor(internal_np)
    # torch.save(internal_tensor, r"C:/Users/Administrator/Desktop/GFM/pdb_data/internal_data/x0s.pt")

    # ================================ internal to cartesian ===================================== #
    # === #
    # internal_path = r"C:/Users/Administrator/Desktop/GFM/traj/linear_internal.pt"                          # Internal coordinate data path
    # internal_tensor = torch.load(internal_path)
    internal_coords = internal_tensor.reshape(-1, 498)
    internal_coords = jnp.array(internal_coords)
    # === #

    cartesian_coords_ex = transformer.to_cartesian(internal_coords)

    cartesian_np = np.array(cartesian_coords_ex)
    cartesian_tensor = torch.tensor(cartesian_np)
    # cartesian_tensor = cartesian_tensor.reshape(internal_tensor.shape[0], -1, 498)
    cartesian_tensor = cartesian_tensor.reshape(1000, -1, 498).cpu().numpy()
    # torch.save(cartesian_tensor, r"C:/Users/Administrator/Desktop/GFM/traj/internal.pt")                     # Output
    ref_pdb = md.load(topology_file)

    output_dir = r"E:\chignolin_results\paths"
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(cartesian_tensor.shape[1]), desc=f'Save paths:'):
        path_coords = cartesian_tensor[:, i, :].reshape(-1, 166, 3)
        traj_instance = md.Trajectory(path_coords, ref_pdb.topology)
        pdb_filename = os.path.join(output_dir, f'path_{i + 1}.pdb')
        traj_instance.save(pdb_filename)

    print(f"所有PDB文件已保存到: {output_dir}")
