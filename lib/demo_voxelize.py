# from mpl_toolkits.mplot3d import Axes3D

from lib.read_mesh import *
from lib.edit_mesh import sample_faces, voxelize_vertices, voxelize_model
from lib.visualize_mesh import visualize_points, visualize_voxels, \
    visualize_rendering
from lib.render_mesh import init_render, render_views

infile = ("data/model.obj")
file3ds = ("data/model.3ds")

if not file_exists(infile):
    print "Couldn't find [%s]" % infile

# Render the mesh
renderer = init_render(256, 256, [file3ds])
renderings = render_views(renderer, 0, [30], [40])

for rendering in renderings:
    visualize_rendering(rendering)


# parse OBJ / MTL files
faces, vertices, uvs, normals, materials, mtllib = parse_obj(infile)
sfaces = sort_faces(faces)

# Visualize vertices
vertices = np.array(vertices)

# Convert the vertex indices into 0-base indices
faces = np.array([x['vertex'] for x in sfaces['triangles_smooth_uv']]) - 1

# Sample vertex
visualize_points(vertices)

sampled_vertices = sample_faces(vertices, faces, 20000)

visualize_points(sampled_vertices)

n_vox = 32

voxels = voxelize_vertices(sampled_vertices, n_vox)

visualize_voxels(voxels)

voxels2 = voxelize_model(infile, n_vox)

visualize_voxels(voxels2)
