# Generate a tetrahedral FSI pipe (lumen and wall) in the svMultiPhysics
# mesh-complete format, matching tests/cases/fsi/pipe_3d (radius 1, wall
# thickness 0.2, length 10). The cross-section is one structured grid: the
# inner square is mapped to the lumen disk and the outer ring of cells to the
# wall annulus, so that the fluid and solid triangulations of the interface
# are identical. Usage: make_fsi_pipe.py <output dir> <cells across the lumen> [<wall layers>]
import os
import sys
import numpy as np

out = sys.argv[1]
nc = int(sys.argv[2])          # cells across the lumen (even)
w = int(sys.argv[3]) if len(sys.argv) > 3 else 2
R, T, L = 1.0, 0.2, 10.0
c = nc // 2
h = 2 * R / nc
nz = int(round(L / h))

# Grid points (a, b) in [-(c+w), c+w]^2, k in [0, nz]
ab = np.arange(-(c + w), c + w + 1)
A, B = np.meshgrid(ab, ab, indexing='ij')
s = np.maximum(np.abs(A), np.abs(B)).astype(float)
def ellip(x, y):
    return x * np.sqrt(1 - y**2 / 2), y * np.sqrt(1 - x**2 / 2)
inner = s <= c
X = np.zeros(A.shape); Y = np.zeros(A.shape)
xi, yi = ellip(A[inner] / c, B[inner] / c)
X[inner], Y[inner] = R * xi, R * yi
ring = ~inner
ux, uy = ellip(A[ring] / s[ring], B[ring] / s[ring])
rr = R + (s[ring] - c) / w * T
X[ring], Y[ring] = rr * ux, rr * uy
npl = A.size
z = np.linspace(0.0, L, nz + 1)
pts = np.zeros((npl, nz + 1, 3))
pts[:, :, 0] = X.ravel()[:, None]
pts[:, :, 1] = Y.ravel()[:, None]
pts[:, :, 2] = z[None, :]
pts = pts.reshape(-1, 3)
ng = 2 * (c + w) + 1
def nid(i, j, k):
    return (i * ng + j) * (nz + 1) + k

# Cells, split into 6 tetrahedra along the main diagonal
I, J, K = np.meshgrid(np.arange(ng - 1), np.arange(ng - 1), np.arange(nz), indexing='ij')
I, J, K = I.ravel(), J.ravel(), K.ravel()
solidCell = np.maximum(np.abs(I - (c + w) + 0.5), np.abs(J - (c + w) + 0.5)) > c
cn = [nid(I + di, J + dj, K + dk) for dk in (0, 1) for dj in (0, 1) for di in (0, 1)]
paths = [(1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6)]
tets = np.concatenate([np.stack([cn[0], cn[a], cn[b], cn[7]], axis=1) for a, b in paths])
isSolid = np.tile(solidCell, 6)
p = pts[tets]
det = np.einsum('ij,ij->i', np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
neg = det < 0
tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()

def boundary_faces(t):
    fl = np.concatenate([t[:, [0, 2, 1]], t[:, [0, 1, 3]], t[:, [1, 2, 3]], t[:, [0, 3, 2]]])
    owner = np.tile(np.arange(len(t)), 4)
    key = np.sort(fl, axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    b = idx[cnt == 1]
    return fl[b], owner[b]

def write_vtu(fn, P, Tt):
    with open(fn, 'w') as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('<UnstructuredGrid>\n<Piece NumberOfPoints="%d" NumberOfCells="%d">\n' % (len(P), len(Tt)))
        f.write('<PointData Scalars="GlobalNodeID">\n<DataArray type="Int32" Name="GlobalNodeID" format="ascii">\n')
        np.savetxt(f, np.arange(1, len(P) + 1), fmt='%d')
        f.write('</DataArray>\n</PointData>\n<CellData Scalars="ModelRegionID">\n')
        f.write('<DataArray type="Int32" Name="ModelRegionID" format="ascii">\n')
        np.savetxt(f, np.ones(len(Tt), int), fmt='%d')
        f.write('</DataArray>\n<DataArray type="Int32" Name="GlobalElementID" format="ascii">\n')
        np.savetxt(f, np.arange(1, len(Tt) + 1), fmt='%d')
        f.write('</DataArray>\n</CellData>\n<Points>\n<DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        np.savetxt(f, P, fmt='%.12g')
        f.write('</DataArray>\n</Points>\n<Cells>\n<DataArray type="Int64" Name="connectivity" format="ascii">\n')
        np.savetxt(f, Tt, fmt='%d')
        f.write('</DataArray>\n<DataArray type="Int64" Name="offsets" format="ascii">\n')
        np.savetxt(f, 4 * np.arange(1, len(Tt) + 1), fmt='%d')
        f.write('</DataArray>\n<DataArray type="UInt8" Name="types" format="ascii">\n')
        np.savetxt(f, np.full(len(Tt), 10), fmt='%d')
        f.write('</DataArray>\n</Cells>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n')

def write_vtp(fn, P, faces, owners, faceId):
    nodes, inv = np.unique(faces, return_inverse=True)
    inv = inv.reshape(faces.shape)
    with open(fn, 'w') as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write('<PolyData>\n<Piece NumberOfPoints="%d" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="%d">\n'
                % (len(nodes), len(faces)))
        f.write('<PointData Scalars="GlobalNodeID">\n<DataArray type="Int32" Name="GlobalNodeID" format="ascii">\n')
        np.savetxt(f, nodes + 1, fmt='%d')
        f.write('</DataArray>\n</PointData>\n<CellData Scalars="ModelFaceID">\n')
        f.write('<DataArray type="Int32" Name="GlobalElementID" format="ascii">\n')
        np.savetxt(f, owners + 1, fmt='%d')
        f.write('</DataArray>\n<DataArray type="Int32" Name="ModelFaceID" format="ascii">\n')
        np.savetxt(f, np.full(len(faces), faceId), fmt='%d')
        f.write('</DataArray>\n</CellData>\n<Points>\n<DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        np.savetxt(f, P[nodes], fmt='%.12g')
        f.write('</DataArray>\n</Points>\n<Polys>\n<DataArray type="Int64" Name="connectivity" format="ascii">\n')
        np.savetxt(f, inv, fmt='%d')
        f.write('</DataArray>\n<DataArray type="Int64" Name="offsets" format="ascii">\n')
        np.savetxt(f, 3 * np.arange(1, len(faces) + 1), fmt='%d')
        f.write('</DataArray>\n</Polys>\n</Piece>\n</PolyData>\n</VTKFile>\n')

for name, mask in (('fluid', ~isSolid), ('solid', isSolid)):
    t = tets[mask]
    used = np.unique(t)
    loc = -np.ones(len(pts), int)
    loc[used] = np.arange(len(used))
    P, Tt = pts[used], loc[t]
    d = os.path.join(out, name)
    os.makedirs(os.path.join(d, 'mesh-surfaces'), exist_ok=True)
    write_vtu(os.path.join(d, 'mesh-complete.mesh.vtu'), P, Tt)
    bf, bo = boundary_faces(Tt)
    cz = P[bf][:, :, 2].mean(axis=1)
    cr = np.hypot(P[bf][:, :, 0], P[bf][:, :, 1]).mean(axis=1)
    groups = {'start': cz < 1e-9, 'end': cz > L - 1e-9}
    side = ~(groups['start'] | groups['end'])
    if name == 'fluid':
        groups['interface'] = side
    else:
        groups['interface'] = side & (cr < R + T / 2)
        groups['outside'] = side & (cr >= R + T / 2)
    print(name, 'nodes', len(P), 'tetrahedra', len(Tt), {k: int(v.sum()) for k, v in groups.items()})
    for fid, (g, m) in enumerate(groups.items(), start=1):
        write_vtp(os.path.join(d, 'mesh-surfaces', g + '.vtp'), P, bf[m], bo[m], fid)
