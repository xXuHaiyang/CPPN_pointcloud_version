from tqdm import tqdm
from scipy.spatial import Voronoi, ConvexHull
import numpy as np

def voronoi_volumes(points):
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # opened regions, i.e., infinite volume (border points)
            vol[i] = 0
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    return vol


# NODE: num
# id, x y z, volume
# MPT: num
# id, x y z, volume
# BORDER: num
# id, x y z, volume=0
# NODE_GROUP: group_num
# id, cat_id
# MPT_GROUP: group_num
# id, cat_id
# BORDER_GROUP: group_num
# id, cat_id=0

def format_outprint(split_parts, filename=None):
    """format the split parts to the output format

    Args:
        split_parts: {"inside": np.array, "border": np.array}
                    inside: 
                        = [(x, y, z, output_state, split_index, geo_or_mat), 
                        (x, y, z, output_state, split_index, geo_or_mat),
                        (x, y, z, output_state, split_index, geo_or_mat)...]
                    border:
                        = [(verts, faces, normals, values),
                        (verts, faces, normals, values),
                        (verts, faces, normals, values)...]
            where: 
                split_index is the index of the split the point belongs to
                geo_or_mat is 0(geometry points) or 1(material points)
        filename (str): filename of the output file

    Writes:
        # NODE: num
        # id, x y z, volume
        # MPT: num
        # id, x y z, volume
        # BORDER: num
        # id, x y z, volume=0
        # NODE_GROUP: group_num
        # id, cat_id
        # MPT_GROUP: group_num
        # id, cat_id
        # BORDER_GROUP: group_num
        # id, cat_id=0
    
    Demo: 
        A cube in the XYZ space
        geometry points and material points are noted as "NODE" and "MPT"
        MPT is split into 2 parts standing for 2 different materials

    """
    print("preprocessing ...")
    inside = split_parts["inside"]
    border = split_parts["border"][0][0]
    
    idx = 1
    flattened_inside = np.concatenate(inside, axis=0).reshape(-1, 6)
    volumes = voronoi_volumes(flattened_inside[:, :3]).reshape(-1, 1)
    points = np.concatenate((flattened_inside, volumes), axis=1)
    # remove err mpt
    err_idx = []
    for i in range(points.shape[0]):
        if points[i, 5] == 1 and points[i, 6] == 0:
            err_idx.append(i)
    err_idx = np.array(err_idx)
    points = np.delete(points, err_idx.astype(int), axis=0)
    
    NODE = points[points[:, 5] == 0] # geometry points
    MPT = points[points[:, 5] == 1]  # material points
    BORDER = border
    
    # NODE: (x, y, z, output_state, split_index, geo_or_mat, volumns)
    # MPT: (x, y, z, output_state, split_index, geo_or_mat, volumns)
    # BORDER: (x, y, z)
    print("start format output ...")
    
    if filename is None:
        filename = str(len(split_parts["inside"])) + "material_" + str(len(NODE)) + \
            "node_" + str(len(MPT)) + "mpt_" + str(len(BORDER)) + "b" + ".cppn"

    with open(filename, 'w') as f:

        # NODE: num
        # id, x y z, volume
        # MPT: num
        # id, x y z, volume
        # BORDER: num
        # id, x y z, volume=0

        node_idx = idx
        f.write("NODE: {}\n".format(len(NODE)))
        print("NODE: {}".format(len(NODE)))
        for node in tqdm(NODE):
            f.write("{} {} {} {} {}\n".format(idx, node[0], node[1], node[2], node[6]))
            idx += 1
        mpt_idx = idx
        f.write("MPT: {}\n".format(len(MPT)))
        print("MPT: {}".format(len(MPT)))
        for mpt in tqdm(MPT):
            f.write("{} {} {} {} {}\n".format(idx, mpt[0], mpt[1], mpt[2], mpt[6]))
            idx += 1
        border_idx = idx
        f.write("BORDER: {}\n".format(len(BORDER)))
        print("BORDER: {}".format(len(BORDER)))
        for border in tqdm(BORDER):
            f.write("{} {} {} {} {}\n".format(idx, border[0], border[1], border[2], 0))
            idx += 1
            
        # NODE_GROUP: group_num
        # id, cat_id
        # MPT_GROUP: group_num
        # id
        # BORDER_GROUP: group_num
        # id, cat_id=0
        
        f.write("NODE_GROUP: {}\n".format(len(split_parts["inside"])))
        print("NODE_GROUP: {}".format(len(split_parts["inside"])))
        for node in tqdm(NODE):
            f.write("{}\n".format(node_idx))
            node_idx += 1
        # f.write("MPT_GROUP: {}\n".format(len(split_parts["inside"])))
        # print("MPT_GROUP: {}".format(len(split_parts["inside"])))
        # for mpt in tqdm(MPT):
        #     f.write("{} {}\n".format(mpt_idx, mpt[4].astype(int)))
        #     mpt_idx += 1
        # f.write("BORDER_GROUP: {}\n".format(len(split_parts["inside"])))
        # print("BORDER_GROUP: {}".format(len(split_parts["inside"])))
        # for border in tqdm(BORDER):
        #     f.write("{} {}\n".format(border_idx, 0))
        #     border_idx += 1

    print("output file {} generated".format(filename))
    return None


# NODE: num
# id, x y z, volume
# MPT: num
# id, x y z, volume
# BORDER: num
# id, x y z, volume=0
# NODE_GROUP: group_num
# id