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
# MPT_GROUP: group_num
# id, cat_id
# SUPPORT_FIXED: num
# id
# LOAD_CONSTANT: num
# id, F_x, F_y, F_z (1, 0, 0)

def format_outprint(split_parts, filename):
    """format the split parts to the output format

    Args:
        split_parts = [part1(np.array), part2, part3...]
            = [(x, y, z, output_state, split_index, geo_or_mat),
            (x, y, z, output_state, split_index, geo_or_mat),
            (x, y, z, output_state, split_index, geo_or_mat)...]
             -- split_index: index of the split the point belongs to
             -- geo_or_mat:  0(geometry points) or 1(material points)
        filename (str): filename of the output file

    Writes:
        NODE: num                      # number of geometry points
        -- id, x, y, z, volume         # index of the point; position of the point; volume of the point
        MPT: num                       # number of material points
        MPT_GROUP: group_num           # number of split parts
        -- id, cat_id                  # index of the point; index of the split part
        SUPPORT_FIXED: num             # number of fixed supports
        -- id                          # index of the point
        LOAD_CONSTANT: num             # number of constant loads
        -- id, F_x, F_y, F_z (1, 0, 0) # index of the point; force in x, y, z direction
    
    Demo: 
        A cube in the XYZ space
        geometry points and material points are noted as "NODE" and "MPT"
        MPT is split into 2 parts standing for 2 different materials
        "SUPPORT_FIXED" at the bottom and "LOAD_CONSTANT" at the top (along the axis of Z)
        "LOAD_CONSTANT" points are influenced by a constant force of 1 in the axis of X
    """
    print("preprocessing ...")
    idx = 1
    flattened_split_parts = np.concatenate(split_parts, axis=0).reshape(-1, 6)
    volumes = voronoi_volumes(flattened_split_parts[:, :3])
    points = np.concatenate((flattened_split_parts, volumes.reshape(-1, 1)), axis=1)
    # points: (x, y, z, output_state, split_index, geo_or_mat, volumns)
    print("start format output ...")
    with open(filename, 'w') as f:
        NODE = points[points[:, 5] == 0] # geometry points
        MPT = points[points[:, 5] == 1]  # material points
        SUPPORT_FIXED = [] # fixed support
        LOAD_CONSTANT = [] # constant load
        f.write("NODE: {}\n".format(len(NODE)))
        print("NODE: {}".format(len(NODE)))
        for node in tqdm(NODE):
            f.write("{} {} {} {} {}\n".format(idx, node[0], node[1], node[2], node[6]))
            # z-axis is the direction of gravity
            if abs(node[2]+1)<0.1:
                SUPPORT_FIXED.append(idx)
            elif abs(node[2]-1)<0.1:
                LOAD_CONSTANT.append(idx)
            idx += 1
        mpt_idx = idx
        f.write("MPT: {}\n".format(len(MPT)))
        print("MPT: {}".format(len(MPT)))
        for mpt in tqdm(MPT):
            f.write("{} {} {} {} {}\n".format(idx, mpt[0], mpt[1], mpt[2], mpt[6]))
            idx += 1
        f.write("MPT_GROUP: {}\n".format(len(split_parts)))
        print("MPT_GROUP: {}".format(len(split_parts)))
        for mpt in tqdm(MPT):
            f.write("{} {}\n".format(mpt_idx, int(mpt[4]+1)))
            mpt_idx += 1
        f.write("SUPPORT_FIXED: {}\n".format(len(SUPPORT_FIXED)))
        print("SUPPORT_FIXED: {}".format(len(SUPPORT_FIXED)))
        for spt_idx in tqdm(SUPPORT_FIXED):
            f.write("{}\n".format(spt_idx))
        f.write("LOAD_CONSTANT: {}\n".format(len(LOAD_CONSTANT)))
        print("LOAD_CONSTANT: {}".format(len(LOAD_CONSTANT)))
        for lct_idx in tqdm(LOAD_CONSTANT):
            f.write("{} {} {} {}\n".format(lct_idx, 1, 0, 0)) # (F_x, F_y, F_z)
    print("output file {} generated".format(filename))
    return None