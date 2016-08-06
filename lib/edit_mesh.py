from lib.read_mesh import *
from math import radians


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def sample_faces(vertices, faces, n_samples=10**4):
    """
    Samples point cloud on the surface of the model defined as vectices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.

    Input : vertices n x 3 matrix
            faces    n x 3 matrix
            n_samples positive integer

    Reference :
      [1] http://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
      [2] Barycentric coordinate system

      \begin{align}
        P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
      \end{align}
    """
    vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                         vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))
    face_areas = face_areas / np.sum(face_areas)

    n_samples_per_face = np.floor(n_samples * face_areas)
    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples, ), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc: acc + _n_sample - 1] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]
    P = (1 - np.sqrt(r[:,0:1])) * A + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B + \
        np.sqrt(r[:,0:1]) * r[:,1:] * C
    return np.concatenate((vertices, P), axis=0)


def find_bounding_cuboid(vertices):
    """
    Given vertices, find the bounding cuboid of the vertices

    Parameters:
      vertices - point cloud n x 3 matrix

    Returns:
      p_min - the minimum point, [x_min,y_min,z_min], of the cuboid that
              encloses the input vertices
      p_max - the maximum point, [x_max,y_max,z_max], of the cuboid that
              encloses the input vertices
    """
    p_min = np.inf * np.ones((3,))
    p_max = - np.inf * np.ones((3,))
    for vertex in vertices:
        for i in range(3):
            if p_min[i] > vertex[i]:
                p_min[i] = vertex[i]
            if p_max[i] < vertex[i]:
                p_max[i] = vertex[i]

    return p_min, p_max


def rotate_vertices_yaw(vertices, azimuth_offset):
    """
    N x 3 vertices and rotate by azimuth_offset
    http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    """
    return np.dot(vertices, rotation_matrix([0, 0, 1], azimuth_offset))

def rotate_vertices_roll(vertices, roll_offset):
    return np.dot(vertices, rotation_matrix([1, 0, 0], roll_offset))


def normalize_vertices(vertices):
    """
    Normalize the point cloud to fin inside a 3D unit sphere
    """
    p_min, p_max = find_bounding_cuboid(vertices)
    center = (p_min + p_max)/2.
    radius = np.linalg.norm(p_max - p_min)/2.

    return (vertices - center)/radius


def create_voxel_grid(p_min, p_max, n):
    """
    Define an axis aligned cube that wraps the cuboid with a small margin

    Parameters:
      p_min - the minimum point of a cuboid defined as [x_min, y_min, z_min]
      p_max - the maximum point of a cuboid defined as [x_min, y_min, z_min]
      n - number of discretization per axis

    Returns:
      grids - a 3 n+1 vectors that defines the start, intervals, and the end of the
              grid along x, y, z axes
    """
    grids = []

    for i in range(3):
        i_grid = np.linspace(p_min[i], p_max[i], n+1, endpoint=True)
        grids.append(i_grid)

    return grids


def voxelize_vertices(vertices, n, azimuth_offset=0, roll_offset=0,fill=False,
                      normalize=True):
    """
    Voxelize input point cloud

    Parameters:
      vertices - point cloud
      n        - number of discretization of the voxels
      fill     - fill the point cloud if true, default

    Returns:
        voxel occupancy map of the point cloud
    """
    voxels = np.zeros((n, n, n), dtype=np.uint8)

    # normalize
    if normalize:
        vertices = normalize_vertices(vertices)

    if azimuth_offset:
        vertices = rotate_vertices_yaw(vertices, azimuth_offset)
    if roll_offset:
        vertices = rotate_vertices_roll(vertices, roll_offset)

    def coord_to_index(coord):
        return [int((elem + 1) * n/2) for elem in coord]
    # You can loop over the voxels or the points
    # Loop over points
    for vertex in vertices:
        idx = tuple(coord_to_index(vertex))
        voxels[idx] = 1

    return voxels


def voxelize_model(infile, n_vox, azimuth_offset=0):
    faces, vertices, uvs, normals, materials, mtllib = parse_obj(infile)
    sfaces = sort_faces(faces)
    vertices = np.array(vertices)

    # Convert the vertex indices into 0-base indices
    faces = np.array([x['vertex'] for x in sfaces['triangles_smooth_uv']]) - 1

    # Sample vertex
    sampled_vertices = sample_faces(vertices, faces, 20000)
    voxels = voxelize_vertices(sampled_vertices, n_vox, azimuth_offset)
    return voxels


def voxelize_model_shapenet(infile, n_vox, azimuth_offset=0):
    faces, vertices, uvs, normals, materials, mtllib = parse_obj(infile)
    vertices = np.array(vertices)

    # Convert the vertex indices into 0-base indices
    faces = np.array([x['vertex'] for x in faces]) - 1

    # Sample vertex
    sampled_vertices = sample_faces(vertices, faces, 20000)
    voxels = voxelize_vertices(sampled_vertices, n_vox, azimuth_offset, roll_offset = radians(-90))
    return voxels
