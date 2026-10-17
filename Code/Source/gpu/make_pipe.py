# Generate a tetrahedral pipe mesh (about 1M tetrahedra) in the svMultiPhysics
# mesh-complete format: volume .vtu and inlet/outlet/wall .vtp faces with
# GlobalNodeID, GlobalElementID and ModelFaceID arrays.
import os
import sys
import numpy as np

R = 1.0          # radius (cm)
n = 25           # grid points per side of the cross-section
m = 291          # grid points along the axis
h = 2 * R / (n - 1)
L = h * (m - 1)

# Cross-section: square [-1,1]^2 mapped to the disk (elliptical mapping)
s = np.linspace(-1.0, 1.0, n)
X, Y = np.meshgrid(s, s, indexing='ij')
U = X * np.sqrt(1 - Y**2 / 2) * R
V = Y * np.sqrt(1 - X**2 / 2) * R
z = np.linspace(0.0, L, m)
pts = np.zeros((n, n, m, 3))
pts[..., 0] = U[:, :, None]
pts[..., 1] = V[:, :, None]
pts[..., 2] = z[None, None, :]
pts = pts.reshape(-1, 3)
nid = np.arange(n * n * m).reshape(n, n, m)

# Hexahedra split into 6 tetrahedra along the main diagonal (conforming)
i, j, k = np.meshgrid(np.arange(n - 1), np.arange(n - 1), np.arange(m - 1), indexing='ij')
i, j, k = i.ravel(), j.ravel(), k.ravel()
def v(di, dj, dk):
    return nid[i + di, j + dj, k + dk]
c = [v(0, 0, 0), v(1, 0, 0), v(0, 1, 0), v(1, 1, 0), v(0, 0, 1), v(1, 0, 1), v(0, 1, 1), v(1, 1, 1)]
paths = [(1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6)]   # 0 -> a -> b -> 7
tets = np.concatenate([np.stack([c[0], c[a], c[b], c[7]], axis=1) for a, b in paths])
# Positive orientation
p = pts[tets]
det = np.einsum('ij,ij->i', np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
neg = det < 0
tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()
nT = len(tets)
print('nodes', len(pts), 'tetrahedra', nT, 'length', L)

# Boundary faces: faces of exactly one tetrahedron
fl = np.concatenate([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [1, 2, 3]], tets[:, [0, 3, 2]]])
owner = np.tile(np.arange(nT), 4)
key = np.sort(fl, axis=1)
_, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
bnd = idx[cnt == 1]
bf, bo = fl[bnd], owner[bnd]
cz = pts[bf][:, :, 2].mean(axis=1)
groups = {'inlet': cz < 1e-9 * L + 1e-12, 'outlet': cz > L - 1e-9}
groups['wall'] = ~(groups['inlet'] | groups['outlet'])

def write_vtu(fn):
    with open(fn, 'w') as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('<UnstructuredGrid>\n<Piece NumberOfPoints="%d" NumberOfCells="%d">\n' % (len(pts), nT))
        f.write('<PointData Scalars="GlobalNodeID">\n<DataArray type="Int32" Name="GlobalNodeID" format="ascii">\n')
        np.savetxt(f, np.arange(1, len(pts) + 1), fmt='%d')
        f.write('</DataArray>\n</PointData>\n<CellData Scalars="ModelRegionID">\n')
        f.write('<DataArray type="Int32" Name="ModelRegionID" format="ascii">\n')
        np.savetxt(f, np.ones(nT, int), fmt='%d')
        f.write('</DataArray>\n<DataArray type="Int32" Name="GlobalElementID" format="ascii">\n')
        np.savetxt(f, np.arange(1, nT + 1), fmt='%d')
        f.write('</DataArray>\n</CellData>\n<Points>\n<DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        np.savetxt(f, pts, fmt='%.10g')
        f.write('</DataArray>\n</Points>\n<Cells>\n<DataArray type="Int64" Name="connectivity" format="ascii">\n')
        np.savetxt(f, tets, fmt='%d')
        f.write('</DataArray>\n<DataArray type="Int64" Name="offsets" format="ascii">\n')
        np.savetxt(f, 4 * np.arange(1, nT + 1), fmt='%d')
        f.write('</DataArray>\n<DataArray type="UInt8" Name="types" format="ascii">\n')
        np.savetxt(f, np.full(nT, 10), fmt='%d')
        f.write('</DataArray>\n</Cells>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n')

def write_vtp(fn, faces, owners, faceId):
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
        np.savetxt(f, pts[nodes], fmt='%.10g')
        f.write('</DataArray>\n</Points>\n<Polys>\n<DataArray type="Int64" Name="connectivity" format="ascii">\n')
        np.savetxt(f, inv, fmt='%d')
        f.write('</DataArray>\n<DataArray type="Int64" Name="offsets" format="ascii">\n')
        np.savetxt(f, 3 * np.arange(1, len(faces) + 1), fmt='%d')
        f.write('</DataArray>\n</Polys>\n</Piece>\n</PolyData>\n</VTKFile>\n')

out = sys.argv[1]
os.makedirs(os.path.join(out, 'mesh-surfaces'), exist_ok=True)
write_vtu(os.path.join(out, 'mesh-complete.mesh.vtu'))
for fid, (name, mask) in enumerate(groups.items(), start=1):
    print(name, mask.sum(), 'triangles')
    write_vtp(os.path.join(out, 'mesh-surfaces', name + '.vtp'), bf[mask], bo[mask], fid)
