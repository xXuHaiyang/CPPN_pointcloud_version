import os
import random
from collections import OrderedDict
from copy import deepcopy

import networkx as nx
import numpy as np
from networkx import DiGraph

from CPPN.NetworkUtils import neg_abs, neg_sqrt_abs, neg_square, normalize, sigmoid, sqrt_abs, neg_sign

class CPPN():

    def __init__(self, output_node_names, num_nodes_to_start=10, num_links_to_start=10):
        self.graph = DiGraph()  # preserving order is necessary for checkpointing
        self.input_node_names = ['x', 'y', 'z', 'd', 'b']
        if type(output_node_names) != list:
            output_node_names = [output_node_names]
        self.output_node_names = output_node_names
        self.activation_functions = [np.sin, np.abs, neg_abs, np.square, neg_square, sqrt_abs, neg_sqrt_abs, np.sign, neg_sign]
        self.set_minimal_graph()  # add input and output nodes
        self.initialize(num_random_node_adds=num_nodes_to_start, num_random_link_adds=num_links_to_start)  # add random nodes and links
        self.morphology = None

    def __str__(self):
        return "CPPN consisting of:\nNodes: " + str(self.graph.nodes()) + "\nEdges: " + str(self.graph.edges())

    def set_minimal_graph(self):
        for name in self.input_node_names:
            self.graph.add_node(name, type="input", function=None)
        for name in self.output_node_names:
            self.graph.add_node(name, type="output", function=sigmoid)
            self.graph.nodes[name]['evaluated'] = False

        for input_node in (node[0] for node in self.graph.nodes(data=True) if node[1]["type"] == "input"):
            for output_node in (node[0] for node in self.graph.nodes(data=True) if node[1]["type"] == "output"):
                self.graph.add_edge(input_node, output_node, weight=0.0)

    def initialize(self, num_random_node_adds=10, num_random_node_removals=0, num_random_link_adds=10,
                   num_random_link_removals=5, num_random_activation_functions=100, num_random_weight_changes=100):

        variation_degree = None
        variation_type = None

        for _ in range(num_random_node_adds):
            variation_degree = self.add_node()
            variation_type = "add_node"

        for _ in range(num_random_node_removals):
            variation_degree = self.remove_node()
            variation_type = "remove_node"

        for _ in range(num_random_link_adds):
            variation_degree = self.add_link()
            variation_type = "add_link"

        for _ in range(num_random_link_removals):
            variation_degree = self.remove_link()
            variation_type = "remove_link"

        for _ in range(num_random_activation_functions):
            variation_degree = self.mutate_function()
            variation_type = "mutate_function"

        for _ in range(num_random_weight_changes):
            variation_degree = self.mutate_weight()
            variation_type = "mutate_weight"

        self.prune_network()
        return variation_type, variation_degree

    def set_input_node_states(self, orig_size_xyz, density=5., load_path="./initial_pointcloud.npy"):
        """Initialize the points cloud randomly and uniformly
        :param1: orig_size_xyz: the whole 3D cube where points cloud could be generated in
        :param2: density: the density of points cloud, i.e., the number of points per unit volume
        :param3: load_path: the path to load the initial point cloud for robot reproduction
        """
        need_initialize = False
        if load_path is None or os.path.exists(load_path) is False: # if the path is not given or the file does not exist
            need_initialize = True
        else:
            save_content = np.load(load_path) 
            # if the file isn't consistent with the given orig_size_xyz and density, we need to re-initialize
            if len(save_content) != int(orig_size_xyz[0] * orig_size_xyz[1] * orig_size_xyz[2] * density): 
                need_initialize = True
        
        if need_initialize:
            flattened_size = int(orig_size_xyz[0] * orig_size_xyz[1] * orig_size_xyz[2] * density)
            input_xyz = np.random.uniform(0, 1, (flattened_size, 3))
            
            for i in range(3):
                input_xyz[:, i] = normalize(input_xyz[:, i]) # [-1, 1]

            input_d = normalize(np.power(np.power(input_xyz[:, 0], 2) + \
                np.power(input_xyz[:, 1], 2) + np.power(input_xyz[:, 2], 2), 0.5))
            input_b = np.ones(flattened_size)
            
            save_path = load_path
            save_content = np.concatenate((input_xyz, input_d.reshape(-1, 1), input_b.reshape(-1, 1)), axis=1)
            np.save(save_path, save_content)
            
        else:
            save_content = np.load(load_path)
            input_xyz = save_content[:, :3]
            input_d = save_content[:, 3]
            input_b = save_content[:, 4]
            
        # index = np.arange(input_xyz.shape[0])
        # self.pointcloud = np.concatenate((input_xyz, index.reshape(-1, 1)), axis=1)
        self.pointcloud = input_xyz

        for name in self.graph.nodes():
            if name == "x":
                self.graph.nodes[name]["state"] = input_xyz[:, 0]
                self.graph.nodes[name]["evaluated"] = True
            if name == "y":
                self.graph.nodes[name]["state"] = input_xyz[:, 1]
                self.graph.nodes[name]["evaluated"] = True
            if name == "z":
                self.graph.nodes[name]["state"] = input_xyz[:, 2]
                self.graph.nodes[name]["evaluated"] = True
            if name == "d":
                self.graph.nodes[name]["state"] = input_d
                self.graph.nodes[name]["evaluated"] = True
            if name == "b":
                self.graph.nodes[name]["state"] = input_b
                self.graph.nodes[name]["evaluated"] = True
    
    def mutate(self):
        """
        selects one mutation type to preform and executes it.
        :return: (variation_degree, str: variation_type)
        """
        mut_type = random.randrange(6)
        variation_degree = None
        variation_type = None

        if mut_type == 0:
            variation_degree = self.add_node()
            variation_type = "add_node"

        elif mut_type == 1:
            variation_degree = self.remove_node()
            variation_type = "remove_node"

        elif mut_type == 2:
            variation_degree = self.add_link()
            variation_type = "add_link"

        elif mut_type == 3:
            variation_degree = self.remove_link()
            variation_type = "remove_link"

        elif mut_type == 4:
            variation_degree = self.mutate_function()
            variation_type = "mutate_function"

        elif mut_type == 5:
            variation_degree = self.mutate_weight()
            variation_type = "mutate_weight"

        self.prune_network()
        return variation_type, variation_degree

    ###############################################
    #   Mutation functions
    ###############################################

    def add_node(self):
        # choose two random nodes (between which a link could exist)
        
        if len(self.graph.edges()) == 0:
            return "NoEdges"

        node1, node2 = random.choice(list(self.graph.edges()))

        # create a new node hanging from the previous output node
        new_node_index = self.get_max_hidden_node_index()
        self.graph.add_node(new_node_index, type="hidden", function=random.choice(self.activation_functions))
        self.graph.nodes[new_node_index]["evaluated"] = False

        # random activation function here to solve the problem with admissible mutations in the first generations
        self.graph.add_edge(new_node_index, node2, weight=1.0)

        # if this edge already existed here, remove it
        # but use it's weight to minimize disruption when connecting to the previous input node
        if (node1, node2) in self.graph.edges():
            weight = self.graph.get_edge_data(node1, node2, default={"weight": 0})["weight"]
            self.graph.remove_edge(node1, node2)
            self.graph.add_edge(node1, new_node_index, weight=weight)
        else:
            self.graph.add_edge(node1, new_node_index, weight=1.0)
            # weight 0.0 would minimize disruption of new edge
            # but weight 1.0 should help in finding admissible mutations in the first generations
        return ""

    def remove_node(self):
        hidden_nodes = list(set(self.graph.nodes(data=False)) - set(self.input_node_names) - set(self.output_node_names))
        if len(hidden_nodes) == 0:
            return "NoHiddenNodes"
        this_node = random.choice(hidden_nodes)

        # if there are edge paths going through this node, keep them connected to minimize disruption
        incoming_edges = self.graph.in_edges(nbunch=[this_node])
        outgoing_edges = self.graph.out_edges(nbunch=[this_node])

        for incoming_edge in incoming_edges:
            for outgoing_edge in outgoing_edges:
                w = self.graph.get_edge_data(incoming_edge[0], this_node, default={"weight": 0})["weight"] * \
                    self.graph.get_edge_data(this_node, outgoing_edge[1], default={"weight": 0})["weight"]
                self.graph.add_edge(incoming_edge[0], outgoing_edge[1], weight=w)

        self.graph.remove_node(this_node)
        return ""

    def add_link(self):
        done = False
        attempt = 0
        while not done:
            done = True

            # choose two random nodes (between which a link could exist, *but doesn't*)
            node1 = random.choice(list(self.graph.nodes()))
            node2 = random.choice(list(self.graph.nodes()))
            while (not self.new_edge_is_valid(node1, node2)) and attempt < 999:
                node1 = random.choice(list(self.graph.nodes()))
                node2 = random.choice(list(self.graph.nodes()))
                attempt += 1
            if attempt > 999:  # no valid edges to add found in 1000 attempts
                done = True

            # create a link between them
            if random.random() > 0.5:
                self.graph.add_edge(node1, node2, weight=0.1)
            else:
                self.graph.add_edge(node1, node2, weight=-0.1)

            # If the link creates a cyclic graph, erase it and try again
            if self.has_cycles():
                self.graph.remove_edge(node1, node2)
                done = False
                attempt += 1
            if attempt > 999:
                done = True
        return ""

    def remove_link(self):
        if len(self.graph.edges()) == 0:
            return "NoEdges"
        this_link = random.choice(list(self.graph.edges()))
        self.graph.remove_edge(this_link[0], this_link[1])
        return ""

    def mutate_function(self):
        this_node = random.choice(list(self.graph.nodes()))
        while this_node in self.input_node_names:
            this_node = random.choice(list(self.graph.nodes()))
        old_function = self.graph.nodes()[this_node]["function"]
        while self.graph.nodes()[this_node]["function"] == old_function:
            self.graph.nodes()[this_node]["function"] = random.choice(self.activation_functions)
        return old_function.__name__ + "-to-" + self.graph.nodes(data=True)[this_node]["function"].__name__

    def mutate_weight(self, mutation_std=0.5):
        if len(self.graph.edges()) == 0:
            return "NoEdges"
        this_edge = random.choice(list(self.graph.edges()))
        node1 = this_edge[0]
        node2 = this_edge[1]
        old_weight = self.graph[node1][node2]["weight"]
        new_weight = old_weight
        while old_weight == new_weight:
            new_weight = random.gauss(old_weight, mutation_std)
            new_weight = max(-1.0, min(new_weight, 1.0))
        self.graph[node1][node2]["weight"] = new_weight
        return float(new_weight - old_weight)

    ###############################################
    #   Helper functions for mutation
    ###############################################

    def prune_network(self):
        """
        Remove erroneous nodes and edges post mutation.
        Recursively removes all such nodes so that every hidden node connects upstream to input nodes and connects downstream to output nodes.
        Removes all hidden nodes that have either no inputs or no outputs.
        """

        done = False
        while not done:
            done = True

            for node in list(self.graph.nodes()):
                in_edge_cnt = len(self.graph.in_edges(nbunch=[node]))
                node_type = self.graph.nodes[node]["type"]

                if in_edge_cnt == 0 and node_type != "input" and node_type != "output":
                    self.graph.remove_node(node)
                    done = False

            for node in list(self.graph.nodes()):
                out_edge_cnt = len(self.graph.out_edges(nbunch=[node]))
                node_type = self.graph.nodes[node]["type"]

                if out_edge_cnt == 0 and node_type != "input" and node_type != "output":
                    self.graph.remove_node(node)
                    done = False

    def has_cycles(self):
        """
        Checks if the graph is a DAG, and returns accordingly.
        :return: True if the graph is not a DAG (has cycles) False otherwise.
        """
        return not nx.is_directed_acyclic_graph(self.graph)

    def get_max_hidden_node_index(self):
        max_index = 0
        for input_node in nx.nodes(self.graph):
            if self.graph.nodes(data=True)[input_node]["type"] == "hidden" and int(input_node) >= max_index:
                max_index = input_node + 1
        return max_index

    def new_edge_is_valid(self, node1, node2):
        """
        Checks that we are permitted to create an edge between these two nodes.
        New edges must:
        * not already exist
        * not create self loops (must be a feed forward network)

        :param node1: Src node
        :param node2: Dest node
        :return: True if an edge can be added. False otherwise.
        """
        if node1 == node2:
            return False
        if node1 in self.output_node_names:
            return False
        if node2 in self.input_node_names:
            return False
        if (node2, node1) in self.graph.edges():
            return False
        if (node1, node2) in self.graph.edges():
            return False
        return True



###############################################
#   For easy testing
###############################################
if __name__ == "__main__":
    
    import open3d as o3d

    # # for test stability, we need to make sure the network is deterministic
    # # when realistcally used, we should eliminate the deterministic part
    random.seed(42)
    np.random.seed(42)
    
    ###############################################
    #  border functions
    ###############################################
    def split(pointcloud, output_state, threshold, border_width=0.1, inside_sparsity=100):
        """split the pointcloud according to the output space

        Args:
            pointcloud (np.array): pointcloud to split
            output_state (np.array): output space of the pointcloud
            threshold (list[float, float, ...]): threshold for the output space to split the pointcloud
            border_width (float, optional): width of the border. Defaults to 0.01.
            inside_sparsity (float, optional): sparsity of the inside points. Defaults to 100.

        Returns:
            split_parts = [part1(np.array), part2, part3...]
            = [(x, y, z, output_state, split_index, geo_or_mat), 
            (x, y, z, output_state, split_index, geo_or_mat),
            (x, y, z, output_state, split_index, geo_or_mat)...]
            where split_index is the index of the split the point belongs to
            geo_or_mat is zero(geometry points) or one(material points)
        """
        if type(threshold) != list:
            threshold = [threshold]
        # whether min/max is valid
        if min(output_state) > max(threshold) or max(output_state) < min(threshold):
            return None
        threshold = [min(output_state)] + threshold + [max(output_state)]
        threshold.sort()
        
        split_parts = []
        for i in range(len(threshold)-1):
            # cube border index
            default_border_idx = np.where(np.logical_or(pointcloud[:, :3] < border_width-1, pointcloud[:, :3] > -border_width+1))[0]
            default_border = np.zeros(pointcloud.shape[0])
            default_border[default_border_idx] = 1
            cube_border = np.logical_and(default_border, np.logical_and(output_state >= threshold[i], output_state <= threshold[i+1]))
            # border points index
            if i == 0: # first split, no floor border
                if_border = np.logical_and(threshold[i+1]-border_width <= output_state, output_state < threshold[i+1])
            elif i == len(threshold)-2: # last split, no ceiling border
                if_border = np.logical_and(threshold[i] <= output_state, output_state <= threshold[i]+border_width)
            else:
                if_border = np.logical_and(threshold[i] <= output_state, output_state <= threshold[i]+border_width) \
                    + np.logical_and(threshold[i+1]-border_width <= output_state, output_state < threshold[i+1])
            if_border = np.logical_or(if_border, cube_border)
            # inside points index
            if i == 0:
                if_inside = output_state < threshold[i+1]-border_width
            elif i == len(threshold)-2:
                if_inside = output_state > threshold[i]+border_width
            else:
                if_inside = np.logical_and(output_state > threshold[i]+border_width, output_state < threshold[i+1]-border_width)
            if_inside = np.logical_and(if_inside, np.logical_not(cube_border))
            
            border_part = pointcloud[if_border]
            inside_part = pointcloud[if_inside][::inside_sparsity]
            output_border_part = output_state[if_border]
            output_inside_part = output_state[if_inside][::inside_sparsity]
            split_border_part = np.ones(len(border_part)) * i
            split_inside_part = np.ones(len(inside_part)) * i
            
            # for border_part, all are geometry points, material_index is zero
            border_geo = np.zeros(len(border_part))
            border = np.concatenate((border_part, output_border_part.reshape(-1, 1), \
                split_border_part.reshape(-1, 1), border_geo.reshape(-1, 1)), axis=1)
            # for inside_part, randomly half are material points, half are geometry points
            inside_geo_or_mat = np.random.randint(0, 2, len(inside_part))
            inside = np.concatenate((inside_part, output_inside_part.reshape(-1, 1), \
                split_inside_part.reshape(-1, 1), inside_geo_or_mat.reshape(-1, 1)), axis=1)
            
            part = np.concatenate((border, inside), axis=0)
            split_parts.append(part)
            
        return split_parts
    
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
    
    shape = (5,5,5)
    density = 5000
    cppn.set_input_node_states(shape, density)
    o = calc_node_state(cppn, "out", shape, density)

    """
    Visualize according to splits
    """
    pts = cppn.pointcloud
    print("total points: ", len(pts))
    split_parts = split(pts, o, [0.5], border_width=0.05, inside_sparsity=100)
    print("parts:", len(split_parts))
    print("total parts points: ", sum([len(part) for part in split_parts]))
    color_parts = [[1, 0, 0], [0, 1, 0], [30/255.,144/255.,1]]
    material_color_parts = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    clouds = []
    for i, split_part in enumerate(split_parts):
        print("part ", str(i+1), ": ", len(split_part))
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(split_part[:, :3])
        colors = np.zeros([split_part.shape[0], 3])
        colors[:, :] = color_parts[i]
        colors[split_part[:, -1] == 1, :] = material_color_parts[i] # material points
        cloud.colors = o3d.utility.Vector3dVector(colors)
        clouds.append(cloud)
        break
    o3d.visualization.draw_geometries(clouds)

    """
    Visualize according to output values
    """
    # pts = cppn.pointcloud
    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(pts[:, :3])
    # colors = np.zeros([pts.shape[0], 3])
    # o_max = np.max(o)
    # o_min = np.min(o)
    # delta_c = abs(o_max - o_min) / (255 * 2)
    # for j in range(pts.shape[0]):
    #     color_n = (o[j] - o_min) / delta_c
    #     if color_n <= 255:
    #         colors[j, :] = [0, 1 - color_n / 255, 1]
    #     else:
    #         colors[j, :] = [(color_n - 255) / 255, 0, 1]
    # cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # # visualize
    # o3d.visualization.draw_geometries([cloud])