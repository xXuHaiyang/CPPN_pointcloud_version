###############################################
#   For easy testing
###############################################
if __name__ == "__main__":
    
    from CPPN import *
    import random
    import numpy as np
    import open3d as o3d

    # # for test stability, we need to make sure the network is deterministic
    # # when realistcally used, we should eliminate the deterministic part
    random.seed(2003)
    np.random.seed(2003)
    
    
    ###############################################
    #   Borrowed from Genome.py for testing
    ###############################################
    def calc_node_state(network, node_name, orig_size_xyz, density=5.):
        """Propagate input values through the network"""
        if network.graph.nodes[node_name]["evaluated"]:
            return network.graph.nodes[node_name]["state"]

        network.graph.nodes[node_name]["evaluated"] = True
        input_edges = network.graph.in_edges(nbunch=[node_name])
        
        flattened_size = int(orig_size_xyz[0] * orig_size_xyz[1] * orig_size_xyz[2] * density)
        new_state = np.zeros(flattened_size)

        for edge in input_edges:
            node1, node2 = edge
            ### recursively evaluate the input node, if it hasn't been evaluated already
            ### at first, all the nodes are un-evaluated except the input nodes, i.e., 'x', 'y', 'z', 'd', 'bias'
            new_state += calc_node_state(network, node1, orig_size_xyz, density) * network.graph.edges[node1, node2]["weight"]

        network.graph.nodes[node_name]["state"] = new_state

        return network.graph.nodes[node_name]["function"](new_state)
    
    cppn = CPPN(output_node_names=["out"])
    
    shape = (2,2,2)
    density = 100
    cppn.set_input_node_states(shape, density)
    o = calc_node_state(cppn, "out", shape, density)
    pts = cppn.pointcloud
    split_parts = geno_to_pheno(pts, o, [0.5], border_width=0.1, inside_sparsity=1)
    format_outprint(split_parts, "test_{}m_{}p.cppn".format(len(split_parts), sum([len(part) for part in split_parts])))
    
    
    """
    o3d Visualize according to splits
    """
    print("total number of points: ", len(pts))
    print("number of materials: ", len(split_parts))
    color_parts = [[1, 0, 0], [0, 1, 0], [30/255.,144/255.,1]]
    material_color_parts = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    cube_color_parts = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    clouds = []
    for i, split_part in enumerate(split_parts):
        print("material ", str(i+1), ": ", len(split_part))
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(split_part[:, :3])
        colors = np.zeros([split_part.shape[0], 3])
        colors[:, :] = color_parts[i]
        colors[split_part[:, -1] == 1, :] = material_color_parts[i] # material points
        colors[split_part[:, -1] == -1, :] = [0, 0, 1] # cube points
        cloud.colors = o3d.utility.Vector3dVector(colors)
        clouds.append(cloud)
        # break
    o3d.visualization.draw_geometries(clouds)
    
    # """
    # A3D Visualize according to splits
    # """
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D 

    # fig = plt.figure()
    # ax = Axes3D(fig)

    # colors = ['r', 'b', 'g']
    # for i, points in enumerate(split_parts):
    #     for t in points:
    #         ax.scatter(t[0], t[1], t[2], c=colors[i], marker='o', s=1)
    # plt.show()

