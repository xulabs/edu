"""
course_toolkit.py — shared helper functions for the BiologyToResearch curriculum
==================================================================================

Every notebook in ``notebooks/`` is *self-contained* (it runs alone on Google
Colab), so the helpers a notebook needs are copied into its "🧰 Course toolkit"
cell.  This file is the single reference copy of those helpers: if you fix a
bug here, copy the fix into the notebooks that use the function
(``python validate_notebooks.py`` lists any copy that differs).

Conventions used everywhere in the course
-----------------------------------------
* Volumes are NumPy arrays indexed ``vol[z, y, x]``; images are ``img[y, x]``.
* The electron beam travels along **z**.  Tilting is a rotation about the **y**
  axis, so the missing wedge lives in the (k_z, k_x) plane of Fourier space.
* Rotation matrices act on physical vectors written in **(x, y, z)** order.
  ``rotate_volume`` converts to array index order internally.
* Euler angles follow the **ZYZ** convention, R = Rz(phi) @ Ry(theta) @ Rz(psi)
  (active rotations, radians), the convention used by AITom/TomoMiner.
* Voxel size of all "molecular zoo" maps is 10 Å = 1 nm, the sampling used by
  most Xu-lab subtomogram papers (Gum-Net uses 1.2 nm, DISCA 1.33 nm).
* SNR (simulation) = var(signal) / var(noise), measured inside the box.
* **Nothing is stored in the repository.**  The molecules of the "zoo" are the real atomic models of the PDB entries
  they are named after, downloaded from the Protein Data Bank (RCSB, with the wwPDB and PDBe as mirrors) the first time a
  notebook needs them and cached outside the repository (``~/.cache/biology_to_research/pdb``; set the environment
  variable BIO2RESEARCH_PDB_CACHE to use another folder).  Without a network connection — or with
  BIO2RESEARCH_STANDINS=1 — every entry falls back to a code-built stand-in of the same size and symmetry, so every
  notebook still runs.  All images, tomograms and other datasets are simulated by code.

Only NumPy and SciPy are required.  Educational code: clarity over speed.
"""
from __future__ import annotations

import io
import os
import struct

import numpy as np
from scipy import ndimage

__version__ = "3.0"

# ── The molecular zoo ──────────────────────────────────────────────────────
# Names, organisms, symmetries and approximate masses (heavy atoms + ~8 % hydrogen) of the PDB entries of the zoo. The
# masses also tell `pdb_atoms` which file to use: the deposited entry (the crystal's asymmetric unit) or the first
# biological assembly, whichever is closer. Like the deposited files, "1bxn" is half of RuBisCO (L4S4) and "1qvr" three
# of the six ClpB subunits; "4v94" holds two TRiC complexes side by side.
ZOO = {
    "4v4r": dict(name="70S ribosome", organism="bacterial (E. coli / T. thermophilus)", mass_kDa=2330, symmetry="C1"),
    "4v7r": dict(name="80S ribosome", organism="S. cerevisiae", mass_kDa=3330, symmetry="C1"),
    "5mrc": dict(name="mitochondrial ribosome", organism="S. cerevisiae", mass_kDa=3320, symmetry="C1"),
    "3j9i": dict(name="20S proteasome", organism="Thermoplasma acidophilum", mass_kDa=710, symmetry="D7"),
    "6utj": dict(name="singly capped proteasome", organism="Thermoplasma acidophilum", mass_kDa=910, symmetry="C7"),
    "4cr2": dict(name="26S proteasome", organism="S. cerevisiae", mass_kDa=1280, symmetry="C1"),
    "4v94": dict(name="TRiC/CCT chaperonin (×2)", organism="S. cerevisiae", mass_kDa=1910, symmetry="C2"),
    "1bxn": dict(name="RuBisCO (L4S4 half)", organism="Cupriavidus necator", mass_kDa=290, symmetry="C4"),
    "1qvr": dict(name="ClpB (3 of 6 subunits)", organism="Thermus thermophilus", mass_kDa=290, symmetry="C1"),
    "3cf3": dict(name="p97/VCP ATPase", organism="Mus musculus", mass_kDa=570, symmetry="C6"),
    "2cg9": dict(name="Hsp90–Sba1 chaperone", organism="S. cerevisiae", mass_kDa=190, symmetry="C2"),
    "1u6g": dict(name="Cand1–Cul1–Roc1 ligase", organism="H. sapiens", mass_kDa=260, symmetry="C1"),
    "3d2f": dict(name="Sse1p–Hsp70 chaperone", organism="S. cerevisiae / H. sapiens", mass_kDa=230, symmetry="C1"),
    "1s3x": dict(name="Hsp70 ATPase domain", organism="H. sapiens", mass_kDa=44, symmetry="C1"),
}
# Other entries used by the course (not part of the zoo of cellular complexes).
_EXTRA_ENTRIES = {"5a1a": dict(name="β-galactosidase", organism="E. coli", mass_kDa=465, symmetry="D2")}


def read_mrc(path_or_bytes) -> tuple[np.ndarray, float]:
    """Minimal MRC2014 reader (modes 0, 1, 2, 6). Returns (array[z, y, x], voxel_size_Å).

    The MRC header is 1024 bytes of little-endian 32-bit words; words 1–3 are the
    grid size (nx, ny, nz), word 4 the data type, words 11–13 the cell size in Å,
    and word 24 the length of an optional extended header.
    """
    raw = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes, "rb").read()
    if raw[:2] == b"\x1f\x8b":  # gzip
        import gzip
        raw = gzip.decompress(raw)
    nx, ny, nz, mode = struct.unpack("<4i", raw[0:16])
    mx, my, mz = struct.unpack("<3i", raw[28:40])
    cella = struct.unpack("<3f", raw[40:52])
    mapc, mapr, maps = struct.unpack("<3i", raw[64:76])
    nsymbt = struct.unpack("<i", raw[92:96])[0]
    dtype = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16}[mode]
    data = np.frombuffer(raw, dtype=dtype, offset=1024 + nsymbt, count=nx * ny * nz)
    vol = data.reshape(nz, ny, nx).astype(np.float32)
    if (mapc, mapr, maps) != (1, 2, 3) and (mapc, mapr, maps) != (0, 0, 0):
        # reorder axes so that the array is [z, y, x]
        order = [3 - maps, 3 - mapr, 3 - mapc]
        vol = np.transpose(vol, np.argsort(order))
    voxel = float(cella[0] / mx) if mx > 0 else 1.0
    return vol, voxel


def write_mrc_bytes(vol: np.ndarray, voxel: float = 1.0) -> bytes:
    """Minimal MRC2014 writer (mode 2, float32). Returns the file contents as bytes (array is [z, y, x])."""
    vol = np.ascontiguousarray(vol, np.float32)
    nz, ny, nx = vol.shape
    hdr = bytearray(1024)
    struct.pack_into("<4i", hdr, 0, nx, ny, nz, 2)
    struct.pack_into("<3i", hdr, 28, nx, ny, nz)
    struct.pack_into("<3f", hdr, 40, nx * voxel, ny * voxel, nz * voxel)
    struct.pack_into("<3f", hdr, 52, 90.0, 90.0, 90.0)
    struct.pack_into("<3i", hdr, 64, 1, 2, 3)
    struct.pack_into("<3f", hdr, 76, float(vol.min()), float(vol.max()), float(vol.mean()))
    hdr[208:212] = b"MAP "
    struct.pack_into("<4B", hdr, 212, 0x44, 0x44, 0, 0)
    struct.pack_into("<f", hdr, 216, float(vol.std()))
    return bytes(hdr) + vol.tobytes()


_BOND = dict(N_CA=1.458, CA_C=1.525, C_N=1.329, C_O=1.231)
_ANGLE = dict(N_CA_C=111.2, CA_C_N=116.2, C_N_CA=121.7, CA_C_O=120.5)
# Side-chain heavy atoms beyond CB (name, element) and natural amino-acid frequencies.
_SIDECHAIN = {
    "GLY": [], "ALA": [], "SER": [("OG", "O")], "CYS": [("SG", "S")], "VAL": [("CG1", "C"), ("CG2", "C")],
    "THR": [("OG1", "O"), ("CG2", "C")], "PRO": [("CG", "C"), ("CD", "C")],
    "LEU": [("CG", "C"), ("CD1", "C"), ("CD2", "C")], "ILE": [("CG1", "C"), ("CG2", "C"), ("CD1", "C")],
    "ASP": [("CG", "C"), ("OD1", "O"), ("OD2", "O")], "ASN": [("CG", "C"), ("OD1", "O"), ("ND2", "N")],
    "GLU": [("CG", "C"), ("CD", "C"), ("OE1", "O"), ("OE2", "O")], "GLN": [("CG", "C"), ("CD", "C"), ("OE1", "O"), ("NE2", "N")],
    "LYS": [("CG", "C"), ("CD", "C"), ("CE", "C"), ("NZ", "N")], "MET": [("CG", "C"), ("SD", "S"), ("CE", "C")],
    "ARG": [("CG", "C"), ("CD", "C"), ("NE", "N"), ("CZ", "C"), ("NH1", "N"), ("NH2", "N")],
    "HIS": [("CG", "C"), ("ND1", "N"), ("CD2", "C"), ("CE1", "C"), ("NE2", "N")],
    "PHE": [("CG", "C"), ("CD1", "C"), ("CD2", "C"), ("CE1", "C"), ("CE2", "C"), ("CZ", "C")],
    "TYR": [("CG", "C"), ("CD1", "C"), ("CD2", "C"), ("CE1", "C"), ("CE2", "C"), ("CZ", "C"), ("OH", "O")],
    "TRP": [("CG", "C"), ("CD1", "C"), ("CD2", "C"), ("NE1", "N"), ("CE2", "C"), ("CE3", "C"), ("CZ2", "C"),
            ("CZ3", "C"), ("CH2", "C")],
}
_AA_FREQ = {"LEU": 9.9, "ALA": 8.3, "GLY": 7.1, "VAL": 6.9, "GLU": 6.8, "SER": 6.6, "ILE": 5.9, "LYS": 5.8, "ARG": 5.5,
            "ASP": 5.5, "THR": 5.3, "PRO": 4.7, "ASN": 4.1, "GLN": 3.9, "PHE": 3.9, "TYR": 2.9, "MET": 2.4, "HIS": 2.3,
            "CYS": 1.4, "TRP": 1.1}


def _nerf(a, b, c, bond, angle, torsion):
    """Place atom d so that |cd| = bond, angle(b,c,d) = angle and dihedral(a,b,c,d) = torsion (degrees)."""
    bc = (c - b) / np.linalg.norm(c - b)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    ang, tor = np.radians(angle), np.radians(torsion)
    d = np.array([-bond * np.cos(ang), bond * np.sin(ang) * np.cos(tor), bond * np.sin(ang) * np.sin(tor)])
    return c + d[0] * bc + d[1] * m + d[2] * n


def build_backbone(phi, psi, omega=None) -> np.ndarray:
    """Backbone atoms N, CA, C, O, CB (array n×5×3, Å) of a chain with the given torsion angles (degrees)."""
    n = len(phi)
    omega = np.full(n, 180.0) if omega is None else np.asarray(omega, float)
    N = [np.zeros(3)]
    CA = [np.array([_BOND["N_CA"], 0.0, 0.0])]
    C = [_nerf(np.array([0.0, 1.0, 0.0]), N[0], CA[0], _BOND["CA_C"], _ANGLE["N_CA_C"], phi[0])]
    for i in range(n - 1):
        N.append(_nerf(N[i], CA[i], C[i], _BOND["C_N"], _ANGLE["CA_C_N"], psi[i]))
        CA.append(_nerf(CA[i], C[i], N[i + 1], _BOND["N_CA"], _ANGLE["C_N_CA"], omega[i]))
        C.append(_nerf(C[i], N[i + 1], CA[i + 1], _BOND["CA_C"], _ANGLE["N_CA_C"], phi[i + 1]))
    N, CA, C = np.array(N), np.array(CA), np.array(C)
    O = np.array([_nerf(N[i], CA[i], C[i], _BOND["C_O"], _ANGLE["CA_C_O"], psi[i] + 180.0) for i in range(n)])
    b, c = CA - N, C - CA
    a = np.cross(b, c)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA      # ideal Cβ position
    return np.stack([N, CA, C, O, CB], 1)


def _ideal_helix_template(n=160):
    """An ideal α-helix (φ = −57.8°, ψ = −47.0°) with its axis along +z through x = y = 0."""
    bb = build_backbone(np.full(n, -57.8), np.full(n, -47.0))
    ca = bb[:, 1]
    centre = ca.mean(0)
    u = np.linalg.svd(ca - centre)[2][0]
    if np.dot(ca[-1] - ca[0], u) < 0:
        u = -u
    # rotation taking u to z
    v = np.cross(u, [0, 0, 1.0]); s = np.linalg.norm(v); cth = u[2]
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    Rm = np.eye(3) + K + K @ K * ((1 - cth) / (s ** 2 + 1e-12))
    xyz = (bb - centre) @ Rm.T
    axis_xy = xyz[:, 1, :2].mean(0)
    xyz[..., :2] -= axis_xy
    return xyz


_HELIX = _ideal_helix_template()


def _rot_to(u):
    """Rotation matrix taking +z to the unit vector u."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.cross([0, 0, 1.0], u); s = np.linalg.norm(v); cth = u[2]
    if s < 1e-9:
        return np.eye(3) if cth > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - cth) / s ** 2)


def _sidechains(bb, resnames, rng):
    """Approximate side-chain heavy atoms grown outward from Cβ (positions are schematic; counts/elements are right)."""
    names, elems, xyz, resid = [], [], [], []
    ca, cb = bb[:, 1], bb[:, 4]
    out = cb - ca
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    for i, rn in enumerate(resnames):
        sc = _SIDECHAIN[rn]
        if not sc:
            continue
        perp = np.cross(out[i], rng.normal(size=3)); perp /= np.linalg.norm(perp)
        for k, (an, el) in enumerate(sc):
            depth = 1.0 + (k // 2)                               # schematic zig-zag outward
            pos = cb[i] + 1.5 * depth * out[i] + 0.9 * (1 if k % 2 else -1) * perp
            names.append(an); elems.append(el); xyz.append(pos); resid.append(i)
    return names, elems, np.array(xyz).reshape(-1, 3), np.array(resid, int)


def _loop_path(a, b, outward, step=3.8):
    """Cα positions of a loop from Cα `a` to Cα `b`: a circular arc bulging along `outward`, cut into equal chords so
    that every Cα–Cα distance (including the two ends) is exactly `step` Å."""
    c = np.linalg.norm(b - a)
    e = (b - a) / c
    u = outward - (outward @ e) * e                                  # bulge direction ⟂ to the chord
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(e, [0.0, 0.0, 1.0]) if abs(e[2]) < 0.9 else np.cross(e, [1.0, 0.0, 0.0])
    u = u / np.linalg.norm(u)
    m = max(int(np.ceil(c / step)), 1)                               # m new residues → n = m + 1 equal chords
    n = m + 1
    lo, hi = 1e-9, 2 * np.pi - 1e-9                                  # arc angle Θ: sin(Θ/2) / sin(Θ/2n) = c / step
    for _ in range(60):
        th = (lo + hi) / 2
        lo, hi = (th, hi) if np.sin(th / 2) / np.sin(th / (2 * n)) > c / step else (lo, th)
    R = step / (2 * np.sin(th / (2 * n)))
    centre = (a + b) / 2 - R * np.cos(th / 2) * u
    beta = np.pi / 2 + th / 2 - th * np.arange(1, n) / n             # angles of the new points, a → b over the bulge
    return centre + R * (np.outer(np.cos(beta), e) + np.outer(np.sin(beta), u))


def _assemble_chain(ca_path_segments, rng, chain="A", start_resnum=1):
    """Turn helices (n×5×3 arrays) joined by loops into one chain with residue names and side chains."""
    blocks = []
    for k, seg in enumerate(ca_path_segments):
        blocks.append(seg)
        if k + 1 < len(ca_path_segments):
            a, b = seg[-1, 1], ca_path_segments[k + 1][0, 1]
            path = _loop_path(a, b, (a + b) / 2 - seg.reshape(-1, 3).mean(0))
            m = len(path)
            tang = np.gradient(path, axis=0) if m > 1 else np.array([b - a])
            tang /= np.linalg.norm(tang, axis=1, keepdims=True)
            side = np.cross(tang, rng.normal(size=3)); side /= np.linalg.norm(side, axis=1, keepdims=True)
            loop = np.stack([path - 1.2 * tang, path, path + 1.2 * tang, path + 1.2 * tang + 1.0 * side,
                             path - 1.3 * side], 1)
            blocks.append(loop)
    bb = np.concatenate(blocks, 0)
    aa = list(_AA_FREQ)
    p = np.array([_AA_FREQ[a] for a in aa]); p /= p.sum()
    resnames = rng.choice(aa, len(bb), p=p)
    bb_names = ["N", "CA", "C", "O", "CB"]
    names, elems, xyz, resid = [], [], [], []
    for i in range(len(bb)):
        for j, an in enumerate(bb_names):
            if an == "CB" and resnames[i] == "GLY":
                continue
            names.append(an); elems.append(an[0]); xyz.append(bb[i, j]); resid.append(i)
    sn, se, sx, sr = _sidechains(bb, resnames, rng)
    names += sn; elems += se; resid = np.concatenate([np.array(resid, int), sr])
    xyz = np.concatenate([np.array(xyz), sx], 0)
    order = np.argsort(resid, kind="stable")
    return dict(xyz=xyz[order].astype(np.float32), name=np.array(names)[order], element=np.array(elems)[order],
                resname=resnames[resid[order]], resnum=(resid[order] + start_resnum).astype(int),
                chain=np.full(len(order), chain))


def helix_bundle(radii_A, rng, chain="A", spacing=10.0):
    """A globular 'domain' that fills an ellipsoid (semi-axes in Å, long axis = local z) with antiparallel helices."""
    a, b, c = radii_A[0] + 2.5, radii_A[1] + 2.5, radii_A[2] + 1.5   # helices stop short of the surface: pad
    rows = []
    j = 0
    while (j * spacing * np.sqrt(3) / 2) <= b:
        rows.append(j); rows.append(-j) if j else None; j += 1
    pts = []
    for jj in sorted(set(rows)):
        y = jj * spacing * np.sqrt(3) / 2
        xs = np.arange(-a, a + 1e-6, spacing) + (spacing / 2 if jj % 2 else 0)
        row = [(x, y) for x in xs if (x / a) ** 2 + (y / b) ** 2 < 1]
        pts.append(row if (jj % 2 == 0) else row[::-1])
    pts = [p for row in pts for p in row]
    segs = []
    for k, (x, y) in enumerate(pts):
        h = c * np.sqrt(max(0.0, 1 - (x / a) ** 2 - (y / b) ** 2)) - 2.0
        n = int(2 * h / 1.5)
        if n < 6:
            continue
        seg = _HELIX[:n].copy()
        seg[..., 2] -= seg[:, 1, 2].mean()
        phase = rng.uniform(0, 2 * np.pi)
        Rz = np.array([[np.cos(phase), -np.sin(phase), 0], [np.sin(phase), np.cos(phase), 0], [0, 0, 1]])
        seg = seg @ Rz.T
        if len(segs) % 2 == 1:                                   # antiparallel neighbours: 180° turn about x
            seg = seg * np.array([1.0, -1.0, -1.0])
        seg = seg + np.array([x, y, rng.uniform(-1.0, 1.0)])
        segs.append(seg)
    return _assemble_chain(segs, rng, chain)


def _transform(atoms, R=np.eye(3), t=(0, 0, 0)):
    out = dict(atoms)
    out["xyz"] = (atoms["xyz"] @ np.asarray(R, float).T + np.asarray(t, float)).astype(np.float32)
    return out


def merge_atoms(parts):
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def _rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def _rx(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _ry(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def atoms_to_pdb_lines(atoms, title="code-generated model (BiologyToResearch course)") -> list:
    """Format an atom dict (xyz, name, element, resname, chain, resnum) as PDB ATOM records (fixed columns)."""
    lines = [f"HEADER    {title[:40]:40s}", "REMARK   1 IDEALISED HELICES BUILT BY CODE - NOT AN EXPERIMENTAL STRUCTURE"]
    for i, (p, an, el, rn, ch, ri) in enumerate(zip(atoms["xyz"], atoms["name"], atoms["element"], atoms["resname"],
                                                    atoms["chain"], atoms["resnum"])):
        name = f" {an:<3s}" if len(an) < 4 else an
        lines.append(f"ATOM  {(i + 1) % 100000:5d} {name:4s} {rn:3s} {ch:1s}{int(ri) % 10000:4d}    "
                     f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00 30.00          {el:>2s}")
    lines.append("END")
    return lines


def _ring(n, radius, z, radii, phase=0.0, tilt=None):
    """n identical subunits (semi-axes `radii` nm: radial, tangential, axial) around the z axis."""
    out = []
    for k in range(n):
        a = phase + 2 * np.pi * k / n
        R = _rz(a) if tilt is None else _rz(a) @ tilt
        out.append(dict(radii=radii, centre=(radius * np.cos(a), radius * np.sin(a), z), R=R, key=("ring", radii, tilt is None)))
    return out


def _lobe(radii, centre, R=None):
    return [dict(radii=radii, centre=centre, R=np.eye(3) if R is None else R, key=None)]


def _dihedral(subunits):
    """Add the partners generated by a 2-fold axis along x (dihedral symmetry)."""
    Rx = _rx(np.pi)
    return subunits + [dict(radii=s["radii"], centre=tuple(Rx @ np.array(s["centre"])), R=Rx @ s["R"], key=s["key"])
                       for s in subunits]


def _blueprint(pid):
    """Subunit layout (nm) of the code-generated stand-in for each zoo entry."""
    if pid in ("4v4r", "4v7r", "5mrc"):
        s = {"4v4r": 1.0, "4v7r": 1.13, "5mrc": 1.14}[pid]
        L = [_lobe((9.0 * s, 8.0 * s, 6.5 * s), (0, 0, 4.5 * s)),                       # large subunit body
             _lobe((3.0 * s, 3.0 * s, 3.0 * s), (0, 1.0 * s, 11.0 * s)),                # central protuberance
             _lobe((2.0 * s, 2.0 * s, 3.5 * s), (-9.0 * s, 0, 7.0 * s), _ry(-0.6)),     # L1-like stalk
             _lobe((1.4 * s, 1.4 * s, 4.0 * s), (9.5 * s, 0, 6.5 * s), _ry(0.7)),       # L7/L12-like stalk
             _lobe((7.5 * s, 5.0 * s, 3.8 * s), (1.0 * s, 0, -4.5 * s)),                # small subunit body
             _lobe((3.5 * s, 3.2 * s, 3.0 * s), (-5.5 * s, -1.0 * s, -6.5 * s))]        # small subunit head
        if pid == "4v7r":                                                                # eukaryotic expansion segments
            L += [_lobe((3.2, 2.6, 2.8), (8.0, 7.0, 7.0)), _lobe((3.0, 2.6, 3.2), (-6.0, -8.0, -6.5)),
                  _lobe((2.4, 2.4, 3.6), (-10.5, 4.0, -1.0), _ry(0.5))]
        if pid == "5mrc":                                                                # mitochondria-specific proteins
            L += [_lobe((4.2, 3.6, 2.8), (0, -7.5, 12.5)), _lobe((3.4, 3.0, 3.4), (9.0, -6.0, -8.0)),
                  _lobe((3.0, 3.0, 3.0), (-9.5, 7.0, 1.0)), _lobe((2.5, 4.5, 2.0), (2.0, 9.0, 9.0))]
        prng = np.random.default_rng(_seed(pid) + 1)                                     # rRNA-like surface bumps
        for _ in range(8):
            d = prng.normal(size=3); d /= np.linalg.norm(d)
            L += [_lobe(tuple(prng.uniform(1.2, 1.9, 3)), tuple(s * d * np.array([9.5, 8.0, 10.5])), _rz(prng.uniform(0, 6)))]
        return [x for l in L for x in l]
    if pid in ("3j9i", "6utj", "4cr2"):
        core = []
        for z, radii, ph in [(1.9, (1.9, 1.7, 2.0), 0.0), (5.6, (2.1, 1.7, 1.85), np.pi / 7)]:
            core += _ring(7, 3.6, z, radii, ph)
        core = _dihedral(core)                                                           # α7β7β7α7, D7
        if pid == "6utj":
            core += _ring(7, 3.3, 10.0, (1.9, 1.6, 2.4), np.pi / 14)                     # one heptameric cap: C7
        if pid == "4cr2":
            core += _ring(6, 4.3, 9.6, (2.2, 1.8, 2.0), 0.3)                             # base ring of six
            core += (_lobe((4.0, 3.0, 3.0), (3.5, 0, 14.5)) + _lobe((3.0, 2.5, 3.5), (-1.5, 3.0, 15.5))
                     + _lobe((2.5, 2.5, 2.5), (-3.5, -2.0, 13.5)) + _lobe((3.0, 3.0, 2.5), (-4.5, 0.5, 10.5)))
        return core
    if pid == "4v94":
        one = []
        for z in (3.6, -3.6):
            one += _ring(8, 5.2, z, (2.6, 2.0, 3.4), np.pi / 8 if z < 0 else 0.0)
        two = []
        for dx, rot in [(-8.5, 0.0), (8.5, np.pi)]:
            for s in one:
                c = _rz(rot) @ np.array(s["centre"])
                two.append(dict(radii=s["radii"], centre=(c[0] + dx, c[1], c[2]), R=_rz(rot) @ s["R"], key=s["key"]))
        return two
    if pid == "1bxn":
        # like the deposited file (the crystal's asymmetric unit): one L4S4 layer = half of the L8S8 enzyme, C4.
        # `_dihedral` of this list would give the complete enzyme (D4).
        return _ring(4, 3.6, 2.2, (2.6, 2.4, 2.4)) + _ring(4, 2.6, 5.0, (1.9, 1.7, 1.35), np.pi / 4)
    if pid == "1qvr":
        # like the deposited file: three neighbouring subunits (half of the C6 hexameric ring), C1
        tiers = [_ring(6, 4.2, 2.2, (2.2, 2.0, 2.0)), _ring(6, 3.9, -2.2, (2.3, 2.0, 2.0), np.pi / 6),
                 _ring(6, 7.2, 2.4, (0.9, 0.9, 4.6), 0.4, tilt=_rx(1.2)),               # coiled-coil 'propellers'
                 _ring(6, 3.8, 6.6, (1.7, 1.6, 1.8), 0.5)]                               # N-domains on top
        return [s for t in tiers for s in t[:3]]
    if pid == "3cf3":
        return (_ring(6, 4.0, 2.0, (2.1, 2.0, 2.0)) + _ring(6, 4.0, -2.2, (2.2, 2.0, 2.1), np.pi / 6)
                + _ring(6, 6.6, 3.0, (2.0, 1.6, 1.8), 0.25))                             # N-domains flush: C6
    if pid == "2cg9":
        half = (_lobe((1.9, 1.7, 1.7), (2.2, 0.8, 6.0)) + _lobe((2.2, 1.8, 2.0), (2.6, -0.4, 2.0))
                + _lobe((1.7, 1.5, 1.8), (1.4, 0.3, -2.2)) + _lobe((1.4, 1.2, 1.2), (4.2, 1.6, 7.6)))
        Rz = _rz(np.pi)
        return half + [dict(radii=h["radii"], centre=tuple(Rz @ np.array(h["centre"])), R=Rz @ h["R"], key=None) for h in half]
    if pid == "1u6g":
        out = []
        for k in range(6):                                                                # curved cullin stalk
            t = k / 5
            out += _lobe((1.4, 1.4, 1.6), (8.0 * np.sin(0.9 * t) - 3.0, 0, 9.0 * t - 4.5), _ry(0.9 * t))
        out += _lobe((2.2, 2.0, 2.0), (4.0, 0, 5.8)) + _lobe((1.2, 1.1, 1.1), (6.0, 1.8, 6.5))
        for k in range(10):                                                               # Cand1-like superhelix
            a = 1.6 * np.pi * k / 9
            out += _lobe((1.5, 1.4, 1.4), (-3.0 + 7.0 * np.sin(0.9 * k / 9) + 2.6 * np.cos(a), 2.6 * np.sin(a), 9.0 * k / 9 - 4.5))
        return out
    if pid == "3d2f":
        comp = (_lobe((1.8, 1.6, 1.8), (0, 0, 0)) + _lobe((1.8, 1.6, 1.8), (0, 3.2, 0)) + _lobe((1.6, 1.4, 2.4), (3.0, 1.6, 1.8))
                + _lobe((1.6, 1.5, 1.6), (-3.2, 1.0, -1.0)) + _lobe((1.6, 1.5, 1.6), (-3.2, 3.6, -0.6)))
        R2, t2 = _rx(2.3) @ _rz(0.7), np.array([1.5, -4.5, 5.5])
        return comp + [dict(radii=c["radii"], centre=tuple(R2 @ np.array(c["centre"]) + t2), R=R2 @ c["R"], key=None) for c in comp]
    if pid == "1s3x":
        return (_lobe((1.4, 1.4, 1.6), (-1.6, 0, -1.2)) + _lobe((1.3, 1.3, 1.5), (-1.4, 0.2, 1.6))
                + _lobe((1.4, 1.4, 1.6), (1.6, 0, -1.2)) + _lobe((1.2, 1.3, 1.4), (1.5, -0.3, 1.8)))
    raise KeyError(pid)


def _seed(pid):
    return int.from_bytes(pid.encode(), "little") % (2 ** 31)


# uniform scale of each blueprint, chosen so that the stand-in's mass is close to the real biological assembly's
_SCALE = {"4v4r": 0.923, "4v7r": 0.906, "5mrc": 0.883, "3j9i": 0.930, "6utj": 0.930, "4cr2": 0.930, "4v94": 0.886,
          "1bxn": 0.930, "1qvr": 0.885, "3cf3": 0.935, "2cg9": 0.995, "1u6g": 0.934, "3d2f": 0.997, "1s3x": 0.884}


def standin_atoms(pid):
    """Atomic model (dict of arrays) of the code-built stand-in for zoo entry `pid` (Å, centred at the origin).

    Used when the real entry cannot be downloaded: ideal α-helices arranged with the size, mass and symmetry of the
    PDB entry. Deterministic (the random numbers are seeded by the entry's name)."""
    pid = pid.lower()
    rng = np.random.default_rng(_seed(pid))
    built, parts = {}, []
    f = _SCALE.get(pid, 1.0)
    for k, s in enumerate(_blueprint(pid)):
        radii_A = tuple(10.0 * f * r for r in s["radii"])
        key = s["key"]
        if key is not None and key in built:
            sub = built[key]
        else:
            sub = helix_bundle(radii_A, rng)
            if key is not None:
                built[key] = sub
        part = _transform(sub, s["R"], 10.0 * f * np.array(s["centre"]))
        part["chain"] = np.full(len(part["xyz"]), _CHAIN_IDS[k % len(_CHAIN_IDS)])
        parts.append(part)
    atoms = merge_atoms(parts)
    if pid in ("3j9i", "6utj", "4cr2"):                             # lay the proteasome barrels along x (side view down z)
        atoms["xyz"] = (atoms["xyz"] @ _ry(np.pi / 2).T).astype(np.float32)
    atoms["xyz"] -= atoms["xyz"].mean(0)
    return atoms


def pdb_cache_dir():
    """Folder (outside the repository) where downloaded PDB files are kept."""
    return os.environ.get("BIO2RESEARCH_PDB_CACHE") or os.path.join(os.path.expanduser("~"), ".cache",
                                                                    "biology_to_research", "pdb")


def _pdb_urls(pid, kind):
    """Download addresses of one PDB file: kind = 'entry' or 'assembly1' (mmCIF) or 'pdb' (legacy PDB format)."""
    pid = pid.lower()
    mid = pid[1:3]
    if kind == "entry":
        return [f"https://files.rcsb.org/download/{pid.upper()}.cif.gz",
                f"https://files.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/{mid}/{pid}.cif.gz",
                f"https://www.ebi.ac.uk/pdbe/entry-files/download/{pid}.cif"]
    if kind == "assembly1":
        return [f"https://files.rcsb.org/download/{pid.upper()}-assembly1.cif.gz",
                f"https://files.wwpdb.org/pub/pdb/data/assemblies/mmCIF/divided/{mid}/{pid}-assembly1.cif.gz"]
    if kind == "pdb":
        return [f"https://files.rcsb.org/download/{pid.upper()}.pdb.gz",
                f"https://files.wwpdb.org/pub/pdb/data/structures/divided/pdb/{mid}/pdb{pid}.ent.gz",
                f"https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{pid}.ent"]
    raise ValueError(kind)


def fetch_pdb(pid, kind="entry", timeout=60):
    """Text of a PDB file, downloaded once and cached outside the repository. Returns (text, where_from), or
    (None, reason) when no mirror can be reached (or BIO2RESEARCH_STANDINS is set)."""
    import gzip
    import urllib.request
    name = {"entry": f"{pid.lower()}.cif.gz", "assembly1": f"{pid.lower()}-assembly1.cif.gz",
            "pdb": f"{pid.lower()}.pdb.gz"}[kind]
    path = os.path.join(pdb_cache_dir(), name)
    if os.path.exists(path):
        return gzip.decompress(open(path, "rb").read()).decode("utf-8", "replace"), f"cached copy ({path})"
    if os.environ.get("BIO2RESEARCH_STANDINS"):
        return None, "BIO2RESEARCH_STANDINS is set"
    reason = "no mirror tried"
    for url in _pdb_urls(pid, kind):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                raw = response.read()
            text = (gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw).decode("utf-8", "replace")
            if ("_atom_site." not in text) and ("\nATOM  " not in text):
                raise ValueError("no atoms in the downloaded file")
            os.makedirs(pdb_cache_dir(), exist_ok=True)
            with open(path, "wb") as f:
                f.write(gzip.compress(text.encode()))
            return text, url
        except Exception as e:  # noqa: BLE001 — any failure: try the next mirror
            reason = f"{type(e).__name__}: {str(e)[:120]}"
    return None, reason


def _cif_tokens(line):
    """Split one line of an mmCIF loop into values (quoted values may contain spaces)."""
    if "'" not in line and '"' not in line:
        return line.split()
    import re
    toks = re.findall(r"'(?:[^']|'(?=\S))*'|\"(?:[^\"]|\"(?=\S))*\"|\S+", line)
    return [t[1:-1] if len(t) > 1 and t[0] == t[-1] and t[0] in "'\"" else t for t in toks]


_WATER = {"HOH", "DOD", "WAT", "H2O"}


def read_mmcif_atoms(text):
    """Atoms of the first model in an mmCIF file → dict(xyz, name, element, resname, chain, resnum).

    Reads the `_atom_site` loop; skips water and hydrogen, and keeps only the first of alternative atom locations."""
    lines = text.splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("_atom_site."))
    cols = []
    while i < len(lines) and lines[i].startswith("_atom_site."):
        cols.append(lines[i].split()[0][len("_atom_site."):])
        i += 1
    c = {name: k for k, name in enumerate(cols)}
    get = lambda *names: next((c[n] for n in names if n in c), None)   # noqa: E731
    ix, iy, iz = c["Cartn_x"], c["Cartn_y"], c["Cartn_z"]
    i_el, i_name = get("type_symbol"), get("auth_atom_id", "label_atom_id")
    i_res, i_ch = get("auth_comp_id", "label_comp_id"), get("auth_asym_id", "label_asym_id")
    i_num, i_alt, i_mod = get("auth_seq_id", "label_seq_id"), get("label_alt_id"), get("pdbx_PDB_model_num")
    xyz, name, element, resname, chain, resnum = [], [], [], [], [], []
    first_model, buf = None, []
    for line in lines[i:]:
        if not line or line.startswith(("#", "_", "loop_", "data_")):
            break
        buf += _cif_tokens(line)
        while len(buf) >= len(cols):
            row, buf = buf[:len(cols)], buf[len(cols):]
            if i_mod is not None:
                first_model = row[i_mod] if first_model is None else first_model
                if row[i_mod] != first_model:
                    continue
            el = row[i_el].upper() if i_el is not None else row[i_name][:1].upper()
            if el in ("H", "D") or row[i_res] in _WATER:
                continue
            if i_alt is not None and row[i_alt] not in (".", "?", "A", "1"):
                continue
            xyz.append((float(row[ix]), float(row[iy]), float(row[iz])))
            name.append(row[i_name])
            element.append(el.capitalize() if len(el) > 1 else el)
            resname.append(row[i_res])
            chain.append(row[i_ch])
            num = row[i_num] if i_num is not None else "0"
            resnum.append(int(num) if num.lstrip("-").isdigit() else 0)
    return dict(xyz=np.array(xyz, np.float32), name=np.array(name), element=np.array(element),
                resname=np.array(resname), chain=np.array(chain), resnum=np.array(resnum))


_ATOMIC_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974, "S": 32.06, "Se": 78.97, "Mg": 24.305,
                "Zn": 65.38, "Fe": 55.845, "Mn": 54.938, "K": 39.098, "Na": 22.99, "Cl": 35.45, "Ca": 40.078}


def model_mass_kDa(atoms):
    """Mass of an atomic model without hydrogens, plus ~8 % for the missing hydrogens (kDa)."""
    return 1.08 * sum(_ATOMIC_MASS.get(e, 12.0) for e in atoms["element"]) / 1000.0


def canonical_frame(atoms):
    """Centre a model and rotate it onto its principal axes — longest extent along x, shortest along z — with the
    signs fixed by the skewness of the atom distribution. A proteasome barrel then lies along x and a ring
    (RuBisCO, p97) lies in the x–y plane, like the stand-ins."""
    xyz = atoms["xyz"].astype(np.float64)
    X = xyz - xyz.mean(0)
    evecs = np.linalg.eigh(np.cov(X.T))[1][:, ::-1]                 # columns: principal axes, largest first
    Y = X @ evecs
    sign = np.sign(np.sum(Y ** 3, axis=0))
    sign[sign == 0] = 1.0
    Y *= sign
    if np.linalg.det(evecs * sign) < 0:                             # keep a proper rotation (no mirror image)
        Y[:, 2] *= -1
    out = dict(atoms)
    out["xyz"] = Y.astype(np.float32)
    return out


def pdb_atoms(pid, verbose=False):
    """Real atomic model of a PDB entry, in the canonical frame. Returns (atoms, description), or (None, reason).

    Uses the deposited entry unless its mass is far from ZOO's reference mass, in which case the first biological
    assembly is tried and the closer of the two kept (e.g. one ribosome rather than the two of a crystal)."""
    pid = pid.lower()
    ref = ZOO.get(pid, {}).get("mass_kDa") or _EXTRA_ENTRIES.get(pid, {}).get("mass_kDa")
    candidates, reason = [], ""
    for kind in ("entry", "assembly1"):
        if candidates and (ref is None or abs(np.log(model_mass_kDa(candidates[0][0]) / ref)) < np.log(1.3)):
            break
        text, where = fetch_pdb(pid, kind)
        if text is None:
            reason = where
            continue
        try:
            atoms = read_mmcif_atoms(text)
        except (StopIteration, KeyError, ValueError) as e:
            reason = f"could not read the file ({type(e).__name__})"
            continue
        if len(atoms["xyz"]):
            label = "deposited entry" if kind == "entry" else "biological assembly 1"
            candidates.append((atoms, f"PDB {pid.upper()} ({label}; {len(atoms['xyz']):,} atoms; from {where})"))
    if not candidates:
        return None, reason
    atoms, desc = min(candidates, key=lambda a: abs(np.log(model_mass_kDa(a[0]) / ref)) if ref else 0.0)
    if verbose:
        print(desc)
    return canonical_frame(atoms), desc


_ATOMS_CACHE = {}


def zoo_atoms(pid):
    """Atomic model (dict of arrays, Å, centred) of zoo entry `pid`: the real PDB model when it can be downloaded, the
    code-built stand-in otherwise. The source is recorded in `atoms_source(pid)`."""
    pid = pid.lower()
    if pid not in _ATOMS_CACHE:
        atoms, desc = pdb_atoms(pid)
        if atoms is None:
            atoms, desc = standin_atoms(pid), f"code-built stand-in modelled on PDB {pid.upper()} (download failed: {desc})"
        _ATOMS_CACHE[pid] = (atoms, desc)
    return _ATOMS_CACHE[pid][0]


def pdb_text_lines(pid):
    """Lines of the legacy-format PDB file of an entry (downloaded once, cached) and their source. Offline, the lines
    are written by code for the stand-in (`atoms_to_pdb_lines`), in the same fixed-column format."""
    text, where = fetch_pdb(pid, "pdb")
    if text is not None:
        return text.splitlines(), f"PDB {pid.upper()} (legacy PDB format, from {where})"
    lines = atoms_to_pdb_lines(standin_atoms(pid), title=f"stand-in for PDB {pid.upper()}")
    return lines, f"code-built stand-in modelled on PDB {pid.upper()} (download failed: {where})"


def atoms_source(pid):
    """Where the atoms of zoo entry `pid` came from (a real PDB file or the offline stand-in)."""
    zoo_atoms(pid)
    return _ATOMS_CACHE[pid.lower()][1]


_Z = {"C": 6.0, "N": 7.0, "O": 8.0, "S": 16.0, "P": 15.0, "H": 1.0, "Mg": 12.0, "Zn": 30.0, "Fe": 26.0, "K": 19.0,
      "Na": 11.0, "Cl": 17.0, "Ca": 20.0, "Mn": 25.0, "Se": 34.0}


def atoms_to_map(xyz, weights=None, voxel=10.0, box=None, sigma_A=None):
    """Density map [z, y, x] from atom positions (Å): trilinear splatting on a grid twice as fine, Gaussian blur
    (σ = `sigma_A`, default = voxel/2), then 2×2×2 binning to `voxel` Å. Centred on the box if `box` is given."""
    xyz = np.asarray(xyz, float)
    w = np.ones(len(xyz)) if weights is None else np.asarray(weights, float)
    fine = voxel / 2.0
    if box is None:
        ext = np.ptp(xyz, 0).max() + 6 * voxel
        box = int(np.ceil(ext / voxel)) + 1
    n = 2 * box
    centre = (n - 1) / 2.0
    p = xyz[:, ::-1] / fine + centre                                  # (z, y, x) grid coordinates
    grid = np.zeros((n, n, n), np.float32)
    i0 = np.floor(p).astype(int)
    f = p - i0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                idx = i0 + np.array([dz, dy, dx])
                wt = w * np.where(dz, f[:, 0], 1 - f[:, 0]) * np.where(dy, f[:, 1], 1 - f[:, 1]) * np.where(dx, f[:, 2], 1 - f[:, 2])
                ok = np.all((idx >= 0) & (idx < n), 1)
                np.add.at(grid, tuple(idx[ok].T), wt[ok])
    grid = ndimage.gaussian_filter(grid, (sigma_A if sigma_A else voxel / 2.0) / fine)
    return grid.reshape(box, 2, box, 2, box, 2).mean((1, 3, 5)).astype(np.float32)


def zoo_map(pid, voxel="10A"):
    """(density[z, y, x], voxel size Å, source) of a zoo entry, painted from its atoms (`zoo_atoms`)."""
    atoms = zoo_atoms(pid)
    v = float(str(voxel).rstrip("A"))
    w = np.array([_Z.get(e, 6.0) for e in atoms["element"]])
    vol = atoms_to_map(atoms["xyz"], w, voxel=v)
    return vol, v, atoms_source(pid)


def betagal_map(voxel=3.54, box=100, b_factor=40.0):
    """E. coli β-galactosidase (PDB 5A1A, a D2 homotetramer, ~4 × 1,020 residues) at `voxel` Å in a `box`³ grid.

    Returns (map[z, y, x], voxel, source, atoms). Atomic B-factor `b_factor` (Å²) blurs each atom by
    σ = √(B / 8π²); helices (≈ 10 Å apart) are visible at this sampling, side chains are not. Offline, a code-built
    β-galactosidase-sized D2 tetramer is used instead.
    """
    atoms, source = pdb_atoms("5a1a")
    if atoms is None:
        atoms, source = _betagal_standin_atoms(), f"code-built β-galactosidase-sized D2 tetramer (download failed: {source})"
    w = np.array([_Z.get(e, 6.0) for e in atoms["element"]])
    sigma = np.sqrt(b_factor / (8 * np.pi ** 2) + (voxel / 2.0) ** 2)                  # B-factor + sampling blur
    vol = atoms_to_map(atoms["xyz"], w, voxel=voxel, box=box, sigma_A=sigma)
    return vol, voxel, source, atoms


def _betagal_standin_atoms():
    """Code-built β-galactosidase-sized D2 homotetramer (chains A–D), centred at the origin."""
    rng = np.random.default_rng(1234)
    domains = [((21.0, 18.0, 22.0), (0.0, 0.0, 0.0)), ((18.0, 16.0, 20.0), (34.0, 8.0, 4.0)),
               ((17.0, 15.0, 19.0), (-31.0, 10.0, -3.0)), ((16.0, 15.0, 18.0), (6.0, 31.0, 8.0)),
               ((15.0, 14.0, 17.0), (-8.0, -29.0, 5.0))]
    parts = [_transform(helix_bundle(r, rng), _rz(rng.uniform(0, 6.3)), c) for r, c in domains]
    mono = merge_atoms(parts)
    mono["xyz"] = mono["xyz"] - mono["xyz"].mean(0) + np.array([30.0, 21.0, 13.0])
    subunits = []
    for k, R in enumerate([np.eye(3), _rz(np.pi), _rx(np.pi), _ry(np.pi)]):          # D2: identity + three 2-folds
        s = _transform(mono, R)
        s["chain"] = np.full(len(s["xyz"]), "ABCD"[k])
        subunits.append(s)
    atoms = merge_atoms(subunits)
    atoms["xyz"] -= atoms["xyz"].mean(0)
    return atoms


def prepare_map(vol: np.ndarray, size: int = 32, sigma: float = 1.0) -> np.ndarray:
    """Low-pass, centre (by centre of mass), crop/pad to ``size``³ and scale to [0, 1].

    Zoo maps are sharp sums of atoms; a Gaussian blur of ~1 voxel (1 nm)
    mimics the resolution of a cellular tomogram.
    """
    v = ndimage.gaussian_filter(np.clip(vol, 0, None).astype(np.float32), sigma)
    s = np.array(v.shape)
    src0 = np.maximum((s - size) // 2, 0)                  # where the crop starts in the input ...
    dst0 = np.maximum((size - s) // 2, 0)                  # ... and where the pad starts in the output
    n = np.minimum(s, size)
    com = np.array(ndimage.center_of_mass(v))
    shift = (size - 1) / 2 - dst0 + src0 - com             # puts the centre of mass exactly at the output's centre
    v = ndimage.shift(v, shift, order=1)
    out = np.zeros((size, size, size), np.float32)
    out[dst0[0]:dst0[0] + n[0], dst0[1]:dst0[1] + n[1], dst0[2]:dst0[2] + n[2]] = \
        v[src0[0]:src0[0] + n[0], src0[1]:src0[1] + n[1], src0[2]:src0[2] + n[2]]
    out -= out.min()
    return out / (out.max() + 1e-12)


def load_zoo(ids=("4v4r", "3j9i", "6utj", "1bxn"), size: int = 32, sigma: float = 1.0,
             verbose: bool = True) -> dict:
    """Build and prepare several zoo maps. Returns {pdb_id: volume(size³)}."""
    zoo = {}
    for pid in ids:
        raw, vs, src = zoo_map(pid)
        zoo[pid] = prepare_map(raw, size=size, sigma=sigma)
        if verbose:
            print(f"  {pid.upper():5s} {ZOO.get(pid, {}).get('name', ''):28s} raw {raw.shape} @ {vs:.0f} Å/voxel  ← {src}")
    return zoo


# ── Rotations ───────────────────────────────────────────────────────────────
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_zyz_to_matrix(phi, theta, psi):
    """R = Rz(phi) · Ry(theta) · Rz(psi)  (radians, acting on (x, y, z) vectors)."""
    return rot_z(phi) @ rot_y(theta) @ rot_z(psi)


def matrix_to_euler_zyz(R):
    """Inverse of :func:`euler_zyz_to_matrix` (one of the two equivalent solutions)."""
    theta = np.arccos(np.clip(R[2, 2], -1, 1))
    if np.isclose(np.sin(theta), 0):
        # Degenerate case (gimbal lock): only a combination of phi and psi is defined, so set psi = 0.
        if R[2, 2] > 0:   # theta = 0:  R = Rz(phi + psi)
            return np.arctan2(R[1, 0], R[0, 0]), theta, 0.0
        # theta = pi: R = Rz(phi - psi) Ry(pi), so R[0, 0] = -cos(phi - psi) and R[1, 0] = -sin(phi - psi)
        return np.arctan2(-R[1, 0], -R[0, 0]), theta, 0.0
    return np.arctan2(R[1, 2], R[0, 2]), theta, np.arctan2(R[2, 1], -R[2, 0])


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random proper rotation (det = +1) from a random unit quaternion."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotation_angle_deg(R1, R2) -> float:
    """Geodesic distance between two rotations: arccos((tr(R1ᵀR2) − 1)/2), in degrees."""
    c = (np.trace(R1.T @ R2) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


_P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float)  # (x,y,z) <-> (z,y,x)


def rotate_volume(vol: np.ndarray, R: np.ndarray, shift_xyz=(0, 0, 0), order: int = 1,
                  cval: float = 0.0) -> np.ndarray:
    """Rotate a volume about its centre by R (then translate by shift, in voxels, (x,y,z)).

    out(p) = vol(Rᵀ (p − c − t) + c): the *object* is rotated by R and moved by t.
    """
    R_idx = _P @ R @ _P                       # same rotation written in (z, y, x) index order
    c = (np.array(vol.shape) - 1) / 2.0
    t_idx = _P @ np.asarray(shift_xyz, float)
    M = R_idx.T                               # inverse rotation maps output -> input
    offset = c - M @ (c + t_idx)
    return ndimage.affine_transform(vol, M, offset=offset, order=order, mode="constant", cval=cval)


def rotate_image(img: np.ndarray, angle_rad: float, shift_xy=(0, 0), order: int = 1) -> np.ndarray:
    """2D analogue of :func:`rotate_volume` (counter-clockwise in the (x, y) plane)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    R_idx = R[::-1, ::-1]                      # (x,y) -> (y,x)
    ctr = (np.array(img.shape) - 1) / 2.0
    t_idx = np.asarray(shift_xy, float)[::-1]
    M = R_idx.T
    return ndimage.affine_transform(img, M, offset=ctr - M @ (ctr + t_idx), order=order, mode="constant")


# ── Fourier space ───────────────────────────────────────────────────────────
def freq_grid(shape):
    """Frequency coordinates (cycles/voxel) for an unshifted FFT; returns list per axis."""
    return np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")


def radial_frequency(shape) -> np.ndarray:
    """|k| in cycles/voxel for every coefficient of an unshifted FFT."""
    return np.sqrt(sum(k**2 for k in freq_grid(shape)))


def missing_wedge_mask(shape, tilt_max_deg: float = 60.0) -> np.ndarray:
    """Binary Fourier mask of a single-axis tilt series about y (±tilt_max).

    A coefficient (k_z, k_y, k_x) is measured if |k_z| ≤ |k_x|·tan(tilt_max)
    — the formula M(ξ) = 1(|ξ_3| ≤ |ξ_1| tan θ) used by Xu et al. (2012).
    For 2D images [z, x] pass a 2-tuple shape.
    """
    k = freq_grid(shape)
    kz, kx = k[0], k[-1]
    m = np.abs(kz) <= np.abs(kx) * np.tan(np.radians(tilt_max_deg)) + 1e-9
    m[(0,) * len(shape)] = True               # keep the mean (DC) term
    return m.astype(np.float32)


def apply_fourier_mask(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Multiply the FFT of ``vol`` by ``mask`` (unshifted layout) and return the real part."""
    return np.real(np.fft.ifftn(np.fft.fftn(vol) * mask)).astype(np.float32)


def lowpass(vol: np.ndarray, cutoff: float, soft: float = 0.02) -> np.ndarray:
    """Soft low-pass filter; ``cutoff`` in cycles/voxel (0.5 = Nyquist)."""
    k = radial_frequency(vol.shape)
    filt = 1 / (1 + np.exp((k - cutoff) / soft))
    return apply_fourier_mask(vol, filt)


def add_noise_snr(signal: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise so that var(signal)/var(noise) = snr."""
    sigma = np.sqrt(signal.var() / snr)
    return (signal + rng.normal(0, sigma, signal.shape)).astype(np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    """Zero mean, unit standard deviation."""
    return ((v - v.mean()) / (v.std() + 1e-12)).astype(np.float32)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two arrays (flattened)."""
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ── Electron optics ─────────────────────────────────────────────────────────
def electron_wavelength_A(kV: float = 300.0) -> float:
    """Relativistic electron wavelength in Å (0.0197 Å at 300 kV)."""
    V = kV * 1e3
    return 12.2643 / np.sqrt(V * (1 + 0.978466e-6 * V))


def ctf_2d(shape, pixel_A: float, defocus_um: float = 3.0, kV: float = 300.0,
           Cs_mm: float = 2.7, amp_contrast: float = 0.07, B: float = 0.0) -> np.ndarray:
    """Contrast transfer function on an unshifted FFT grid (underfocus is positive).

    CTF(k) = −[√(1−A²)·sin χ(k) + A·cos χ(k)]·exp(−B k²/4),
    χ(k) = π λ Δf k² − (π/2) C_s λ³ k⁴, with k in 1/Å.
    """
    lam = electron_wavelength_A(kV)
    k2 = sum(kk**2 for kk in freq_grid(shape)) / pixel_A**2
    dz = defocus_um * 1e4
    cs = Cs_mm * 1e7
    chi = np.pi * lam * dz * k2 - 0.5 * np.pi * cs * lam**3 * k2**2
    ctf = -(np.sqrt(1 - amp_contrast**2) * np.sin(chi) + amp_contrast * np.cos(chi))
    return (ctf * np.exp(-B * k2 / 4)).astype(np.float32)


# ── Tomography ──────────────────────────────────────────────────────────────
def project_tilt_series(vol: np.ndarray, angles_deg) -> np.ndarray:
    """Tilt-series projections of vol[z, y, x] about the y axis → array[n_tilts, y, x]."""
    return np.stack([ndimage.rotate(vol, a, axes=(0, 2), reshape=False, order=1).sum(axis=0)
                     for a in angles_deg]).astype(np.float32)


def weighted_back_projection(tilts: np.ndarray, angles_deg, nz: int) -> np.ndarray:
    """Weighted (ramp-filtered) back-projection of a y-axis tilt series → vol[nz, y, x]."""
    n, ny, nx = tilts.shape
    ramp = np.abs(np.fft.fftfreq(nx))[None, None, :]
    filt = np.real(np.fft.ifft(np.fft.fft(tilts, axis=2) * ramp, axis=2))
    rec = np.zeros((nz, ny, nx), np.float32)
    for img, a in zip(filt, angles_deg):
        smear = np.broadcast_to(img[None], (nz, ny, nx))
        rec += ndimage.rotate(smear, -a, axes=(0, 2), reshape=False, order=1)
    return rec / n


def simulate_subtomogram(template: np.ndarray, rng: np.random.Generator, snr: float = 0.1,
                         tilt_max_deg: float | None = 60.0, max_shift: float = 0.0,
                         R: np.ndarray | None = None):
    """Randomly rotate/shift a template, impose a missing wedge and add noise.

    Returns (subtomogram, R, shift_xyz).  This 'direct' simulation (Fourier mask +
    white noise) is the fast approximation used by many Xu-lab benchmarks.
    """
    R = random_rotation(rng) if R is None else R
    t = rng.uniform(-max_shift, max_shift, 3) if max_shift > 0 else np.zeros(3)
    v = rotate_volume(template, R, t)
    if tilt_max_deg is not None:
        v = apply_fourier_mask(v, missing_wedge_mask(v.shape, tilt_max_deg))
    return add_noise_snr(v, snr, rng), R, t


# ── Validation ──────────────────────────────────────────────────────────────
def fsc(v1: np.ndarray, v2: np.ndarray, pixel_A: float = 10.0, n_shells: int | None = None):
    """Fourier shell (ring, in 2D) correlation. Returns (frequency_1/Å, fsc, n_coeffs)."""
    F1, F2 = np.fft.fftn(v1), np.fft.fftn(v2)
    k = radial_frequency(v1.shape)
    n_shells = n_shells or min(v1.shape) // 2
    idx = np.minimum((k * 2 * n_shells).astype(int), n_shells)
    num = np.bincount(idx.ravel(), (F1 * np.conj(F2)).real.ravel(), n_shells + 1)
    d1 = np.bincount(idx.ravel(), (np.abs(F1)**2).ravel(), n_shells + 1)
    d2 = np.bincount(idx.ravel(), (np.abs(F2)**2).ravel(), n_shells + 1)
    cnt = np.bincount(idx.ravel(), minlength=n_shells + 1)
    curve = num / np.sqrt(d1 * d2 + 1e-20)
    freq = (np.arange(n_shells + 1) + 0.5) / (2 * n_shells) / pixel_A
    return freq[:n_shells], curve[:n_shells], cnt[:n_shells]


def resolution_at(freq, curve, threshold: float = 0.143) -> float:
    """First crossing of the FSC curve below ``threshold`` → resolution in Å."""
    below = np.where(curve[1:] < threshold)[0]
    if len(below) == 0:
        return float(1 / freq[-1])
    i = below[0] + 1
    f0, f1, c0, c1 = freq[i - 1], freq[i], curve[i - 1], curve[i]
    f = f0 + (threshold - c0) * (f1 - f0) / (c1 - c0 + 1e-12)
    return float(1 / f)


# ── Flexible molecules, Gaussian mixtures and reconstruction (Part F) ─────────
def flexible_protein(seed: int = 7):
    """A code-generated ~80 kDa protein for heterogeneity experiments (Modules 25–30).

    Domains 0 and 1 form a rigid body; domain 2 (the 'arm') sits on domain 1 and can swing about a hinge; domain 3 is
    a small partner subunit that may be bound or not. Returns (atoms, hinge (x, y, z) in Å); atoms['domain'] labels
    every atom. Coordinates are in Å, centred near the origin.
    """
    rng = np.random.default_rng(seed)
    specs = [((21.0, 17.5, 16.0), (-19.0, -7.5, 0.0)), ((19.0, 16.0, 17.5), (19.0, -7.5, 0.0)),
             ((17.5, 12.5, 14.0), (21.0, 33.0, 0.0)), ((12.5, 11.0, 11.0), (-21.0, 26.0, 2.5))]
    parts = []
    for d, (radii, centre) in enumerate(specs):
        part = _transform(helix_bundle(radii, rng, chain="ABCD"[d]), _rz(rng.uniform(0, 2 * np.pi)), centre)
        part["domain"] = np.full(len(part["xyz"]), d)
        parts.append(part)
    atoms = merge_atoms(parts)
    shift = atoms["xyz"][atoms["domain"] < 3].mean(0)
    atoms["xyz"] = (atoms["xyz"] - shift).astype(np.float32)
    return atoms, np.array([19.0, 14.0, 0.0]) - shift


def move_domain(atoms, domain: int, R: np.ndarray, pivot) -> dict:
    """Copy of `atoms` with every atom of `domain` rotated by R (acting on (x, y, z)) about `pivot` (Å)."""
    out = dict(atoms)
    xyz = atoms["xyz"].astype(np.float64).copy()
    sel = atoms["domain"] == domain
    xyz[sel] = (xyz[sel] - pivot) @ np.asarray(R, float).T + pivot
    out["xyz"] = xyz.astype(np.float32)
    return out


def pseudoatoms_kmeans(vol: np.ndarray, n: int, voxel: float = 1.0, rng=None, iters: int = 30,
                       level: float = 0.05):
    """Summarise a density map by n Gaussian 'pseudo-atoms': weighted K-means on the voxels above level × max.

    Returns centres (n, 3) as (x, y, z) in Å relative to the box centre, and amplitudes (density in each cluster).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    idx = np.argwhere(vol > level * vol.max())                                 # (z, y, x) of dense voxels
    w = vol[tuple(idx.T)].astype(np.float64)
    pts = ((idx[:, ::-1] - (np.array(vol.shape[::-1]) - 1) / 2.0) * voxel).astype(np.float32)
    c = pts[rng.choice(len(pts), n, replace=False, p=w / w.sum())].copy()
    for _ in range(iters):
        d2 = (pts ** 2).sum(1)[:, None] - 2 * pts @ c.T + (c ** 2).sum(1)[None]
        lab = d2.argmin(1)
        ws = np.bincount(lab, w, n)
        for a in range(3):
            upd = np.bincount(lab, w * pts[:, a], n)
            c[ws > 0, a] = upd[ws > 0] / ws[ws > 0]
    return c.astype(np.float32), np.bincount(lab, w, n).astype(np.float32)


def gmm_to_map(centres, amps, sigmas, box: int, voxel: float = 1.0) -> np.ndarray:
    """Density [z, y, x] (box³, origin at the box centre) of Σ_j A_j exp(−|x − c_j|² / 2σ_j²); c and σ in Å."""
    centres = np.asarray(centres, float)
    s = np.broadcast_to(np.asarray(sigmas, float), (len(centres),))
    g = (np.arange(box) - (box - 1) / 2.0) * voxel
    ex = [np.exp(-(g[None, :] - centres[:, a, None]) ** 2 / (2 * s[:, None] ** 2)) for a in range(3)]
    return np.einsum("n,nz,ny,nx->zyx", np.asarray(amps, float), ex[2], ex[1], ex[0], optimize=True).astype(np.float32)


def gmm_project(centres, amps, sigmas, R, box: int, pixel: float = 1.0) -> np.ndarray:
    """Projection [y, x] along z of a Gaussian mixture rotated by R (acting on (x, y, z)); same units as `gmm_to_map`.

    A 3D Gaussian projects to a 2D Gaussian of the same width, so the image is Σ_j w_j · g_y,j ⊗ g_x,j — two small
    matrices instead of a 3D grid (the 'project x and y separately' trick).
    """
    p = (np.asarray(centres, np.float32) @ np.asarray(R, np.float32).T)
    s = np.broadcast_to(np.asarray(sigmas, np.float32), (len(p),))[:, None]
    g = ((np.arange(box) - (box - 1) / 2.0) * pixel).astype(np.float32)[None, :]
    gx = np.exp(-(g - p[:, 0, None]) ** 2 / (2 * s ** 2))
    gy = np.exp(-(g - p[:, 1, None]) ** 2 / (2 * s ** 2))
    w = np.asarray(amps, np.float32)[:, None] * np.float32(np.sqrt(2 * np.pi)) * s / np.float32(pixel)   # ∫ dz
    return ((gy * w).T @ gx).astype(np.float32)


def reconstruct_fourier(images: np.ndarray, rotations, wiener: float = 1.0, chunk: int = 64) -> np.ndarray:
    """3D map [z, y, x] from projection images [n, y, x] with known rotations (R acts on (x, y, z), as in
    `rotate_volume`; image = rotate_volume(vol, R).sum(axis=0)).

    Projection-slice theorem: each image's 2D Fourier transform is the central slice of the 3D transform at
    k₃ = Rᵀ(kx, ky, 0). Slices are inserted with trilinear weights into a 2× padded grid, divided by the summed
    weights (+ `wiener`, in units of one image), transformed back and corrected for the interpolation kernel.
    """
    images = np.asarray(images, np.float64)
    n, N, _ = images.shape
    M, pad = 2 * N, N // 2
    k = np.arange(M) - M // 2
    ky, kx = np.meshgrid(k, k, indexing="ij")
    keep = (kx ** 2 + ky ** 2).ravel() <= (M // 2 - 1) ** 2
    plane = np.stack([kx.ravel(), ky.ravel(), np.zeros(M * M)], 1)[keep]
    shift_phase = np.exp(-1j * np.pi * (kx + ky) / M).ravel()[keep]          # half-pixel: image centre → index M/2
    Fr, Fi, W = np.zeros(M ** 3), np.zeros(M ** 3), np.zeros(M ** 3)
    corners = [(dz, dy, dx) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
    for start in range(0, n, chunk):
        idx_all, val_all, w_all = [], [], []
        for img, R in zip(images[start:start + chunk], list(rotations)[start:start + chunk]):
            big = np.zeros((M, M)); big[pad:pad + N, pad:pad + N] = img
            Fimg = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(big))).ravel()[keep] * shift_phase
            q = plane @ np.asarray(R, float)                                    # rows: (Rᵀ k)ᵀ in (x, y, z)
            pz = q[:, ::-1] + M // 2                                            # (z, y, x) grid coordinates
            i0 = np.floor(pz).astype(int); f = pz - i0
            for dz, dy, dx in corners:
                ii = i0 + (dz, dy, dx)
                wt = (np.where(dz, f[:, 0], 1 - f[:, 0]) * np.where(dy, f[:, 1], 1 - f[:, 1])
                      * np.where(dx, f[:, 2], 1 - f[:, 2]))
                ok = np.all((ii >= 0) & (ii < M), 1)
                idx_all.append((ii[ok, 0] * M + ii[ok, 1]) * M + ii[ok, 2]); val_all.append(wt[ok] * Fimg[ok])
                w_all.append(wt[ok])
        idx, val, wts = np.concatenate(idx_all), np.concatenate(val_all), np.concatenate(w_all)
        Fr += np.bincount(idx, val.real, M ** 3); Fi += np.bincount(idx, val.imag, M ** 3); W += np.bincount(idx, wts, M ** 3)
    F3 = ((Fr + 1j * Fi) / (W + wiener)).reshape(M, M, M)
    kz3, ky3, kx3 = np.meshgrid(k, k, k, indexing="ij")
    F3 *= np.exp(1j * np.pi * (kx3 + ky3 + kz3) / M)                            # centre back to (N − 1)/2
    vol = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(F3))))[pad:pad + N, pad:pad + N, pad:pad + N]
    x = (np.arange(N) - (N - 1) / 2.0) / M
    sinc2 = np.sinc(x) ** 2                                                     # trilinear-kernel correction
    return (vol / (sinc2[:, None, None] * sinc2[None, :, None] * sinc2[None, None, :])).astype(np.float32)


# ── Metrics ─────────────────────────────────────────────────────────────────
def match_points(pred, true, radius: float):
    """Greedy one-to-one matching of predicted to true coordinates within ``radius``.

    Returns dict(precision, recall, f1, tp, fp, fn).
    """
    true = np.asarray(true, float)
    pred = np.asarray(pred, float).reshape(-1, true.shape[-1] if true.ndim == 2 else 3)
    if len(pred) == 0 or len(true) == 0:
        tp = 0
    else:
        d = np.linalg.norm(pred[:, None, :] - true[None, :, :], axis=-1)
        used_p, used_t, tp = set(), set(), 0
        for i, j in zip(*np.unravel_index(np.argsort(d, axis=None), d.shape)):
            if d[i, j] > radius:
                break
            if i not in used_p and j not in used_t:
                used_p.add(i); used_t.add(j); tp += 1
    fp, fn = len(pred) - tp, len(true) - tp
    p = tp / max(len(pred), 1)
    r = tp / max(len(true), 1)
    return dict(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-12), tp=tp, fp=fp, fn=fn)


def hungarian_accuracy(y_true, y_pred) -> float:
    """Clustering accuracy after the best one-to-one relabelling (Hungarian algorithm)."""
    from scipy.optimize import linear_sum_assignment
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ct = np.zeros((y_pred.max() + 1, y_true.max() + 1))
    np.add.at(ct, (y_pred, y_true), 1)
    r, c = linear_sum_assignment(-ct)
    return float(ct[r, c].sum() / len(y_true))


# ── Crowded "in silico cell" ────────────────────────────────────────────────
def support_radius(t: np.ndarray, level: float = 0.1) -> float:
    """Radius (voxels) of the smallest centred sphere containing all voxels above ``level``·max."""
    pts = np.argwhere(t > level * t.max())
    if len(pts) == 0:
        return t.shape[0] / 2
    c = (np.array(t.shape) - 1) / 2
    return float(np.linalg.norm(pts - c, axis=1).max()) + 1.0


def make_crowded_cell(templates: dict, counts: dict, shape=(48, 160, 160), rng=None,
                      membrane: bool = True, tries_per_particle: int = 400):
    """Pack randomly rotated templates into a volume without overlap (bounding spheres).

    A simple rejection-sampling stand-in for the packing used by Pei et al. (2016)
    and Liu et al. (2020).  Particles are placed in random class order so that every
    class gets a fair chance.  Returns (clean_volume, particles) where each particle
    is dict(cls, center_zyx, R).  Optionally adds a spherical membrane vesicle.
    """
    rng = rng or np.random.default_rng(0)
    vol = np.zeros(shape, np.float32)
    occ_c, occ_r, particles = [], [], []
    if membrane:  # a vesicle: thin shell, ~4 nm thick lipid bilayer at 1 nm/voxel
        zc, yc, xc = shape[0] / 2, shape[1] * 0.78, shape[2] * 0.22
        rad = min(shape[1], shape[2]) * 0.16
        g = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
        d = np.sqrt((g[0] - zc)**2 + (g[1] - yc)**2 + (g[2] - xc)**2)
        vol += 0.6 * np.exp(-((d - rad)**2) / (2 * 1.5**2))
        occ_c.append(np.array([zc, yc, xc])); occ_r.append(rad + 3)
    radii = {cls: support_radius(templates[cls], level=0.35) for cls in counts}
    order = [cls for cls, n in counts.items() for _ in range(n)]
    rng.shuffle(order)
    for cls in order:
        t, rad_t = templates[cls], radii[cls]
        for _ in range(tries_per_particle):
            c = np.array([rng.uniform(rad_t, s - rad_t) if s > 2 * rad_t else s / 2 for s in shape])
            if any(np.linalg.norm(c - oc) < rad_t + orr for oc, orr in zip(occ_c, occ_r)):
                continue
            R = random_rotation(rng)
            sub = rotate_volume(t, R)
            h = np.array(sub.shape) // 2
            lo = np.round(c).astype(int) - h
            sl_v, sl_s = [], []
            for a in range(3):
                a0, a1 = max(lo[a], 0), min(lo[a] + sub.shape[a], shape[a])
                sl_v.append(slice(a0, a1)); sl_s.append(slice(a0 - lo[a], a1 - lo[a]))
            vol[tuple(sl_v)] += sub[tuple(sl_s)]
            occ_c.append(c); occ_r.append(rad_t)
            particles.append(dict(cls=cls, center_zyx=c, R=R))
            break
    return vol, particles


def simulate_tomogram(clean: np.ndarray, rng, snr: float = 0.05, tilt_max_deg: float = 60.0,
                      ctf_pixel_A: float | None = 10.0, defocus_um: float = 5.0) -> np.ndarray:
    """Fast tomogram simulation: (optional 3D CTF-like filter) → missing wedge → noise.

    The CTF is applied isotropically here (a simplification of per-tilt CTF).
    """
    v = clean
    if ctf_pixel_A:
        ctf = ctf_2d(clean.shape, ctf_pixel_A, defocus_um=defocus_um, kV=300, B=0)
        v = apply_fourier_mask(v, -ctf)          # sign flip: dense objects appear bright
    v = apply_fourier_mask(v, missing_wedge_mask(v.shape, tilt_max_deg))
    return add_noise_snr(v, snr, rng)


# ── Plotting ────────────────────────────────────────────────────────────────
def show_orthoslices(vol, title: str = "", axes=None, cmap: str = "gray"):
    """Show the three central orthogonal slices (xy, xz, yz) of a volume."""
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
    z, y, x = (np.array(vol.shape) // 2)
    for ax, im, lab in zip(axes, [vol[z], vol[:, y, :], vol[:, :, x]], ["xy (z mid)", "xz (y mid)", "yz (x mid)"]):
        ax.imshow(im, cmap=cmap)
        ax.set_title(f"{title} {lab}".strip(), fontsize=8)
        ax.axis("off")
    return axes
