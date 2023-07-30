import os
import random
from collections import OrderedDict
from copy import deepcopy

import networkx as nx
import numpy as np
from networkx import DiGraph

from .NetworkUtils import neg_abs, neg_sqrt_abs, neg_square, normalize, sigmoid, sqrt_abs, neg_sign

class CPPN():

    def __init__(self, output_node_names, num_nodes_to_start=10, num_links_to_start=10, mode="random"):
        self.graph = DiGraph()  # preserving order is necessary for checkpointing
        self.input_node_names = ['x', 'y', 'z', 'd', 'b']
        if type(output_node_names) != list:
            output_node_names = [output_node_names]
        self.output_node_names = output_node_names
        self.activation_functions = [np.sin, np.abs, neg_abs, np.square, neg_square, sqrt_abs, neg_sqrt_abs, np.sign, neg_sign]
        self.set_minimal_graph()  # add input and output nodes
        self.initialize(num_random_node_adds=num_nodes_to_start, num_random_link_adds=num_links_to_start)  # add random nodes and links
        self.morphology = None
        self.mode = mode

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
        Args:
            orig_size_xyz: the whole 3D cube where points cloud could be generated in
            density: the density of points cloud, i.e., the number of points per unit volume
            load_path: the path to load the initial point cloud for robot reproduction
        """
        need_initialize = True # NOTE for debug
        
        if load_path is None or os.path.exists(load_path) is False: # if the path is not given or the file does not exist
            need_initialize = True
        else:
            save_content = np.load(load_path) 
            # if the file isn't consistent with the given orig_size_xyz and density, we need to re-initialize
            if self.mode == "random":
                if len(save_content) != int(orig_size_xyz[0] * orig_size_xyz[1] * orig_size_xyz[2] * (density**3)): 
                    need_initialize = True
            else:
                self.num_geos = int(orig_size_xyz[0] * orig_size_xyz[1] * orig_size_xyz[2] * density**3)
                self.num_mats = int((orig_size_xyz[0]*density-1)*(orig_size_xyz[1]*density-1)*(orig_size_xyz[2]*density-1))
                if len(save_content) != self.num_geos + self.num_mats:
                    need_initialize = True
        
        if need_initialize:
            assert self.mode in ["random", "structural"], "mode should be either random or structural"

            if self.mode == "random":
                self.num_points = int(orig_size_xyz[0] * orig_size_xyz[1] * orig_size_xyz[2] * (density**3))
                input_xyz = np.random.uniform(0, 1, (self.num_points, 3))
            else: # np.meshgrid for structured points cloud
                # geometrically, the points cloud is a cube, and the cube is scaled by density
                l, w, h = orig_size_xyz[0] * density, orig_size_xyz[1] * density, orig_size_xyz[2] * density
                x = np.linspace(0, 1, l)
                y = np.linspace(0, 1, w)
                z = np.linspace(0, 1, h)
                xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
                input_xyz_geo = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=1)
                self.num_geos = len(input_xyz_geo)
                
                # materialize the points cloud
                x_ = np.linspace(0, 1, l)
                y_ = np.linspace(0, 1, w)
                z_ = np.linspace(0, 1, h)
                xx_, yy_, zz_ = np.meshgrid(x_, y_, z_, indexing='ij')
                input_xyz_mat = np.stack((xx_.flatten(), yy_.flatten(), zz_.flatten()), axis=1)
                input_xyz_mat[:, 0] = input_xyz_mat[:, 0] + 1/2 * 1 / (l-1)
                input_xyz_mat[:, 1] = input_xyz_mat[:, 1] + 1/2 * 1 / (w-1)
                input_xyz_mat[:, 2] = input_xyz_mat[:, 2] + 1/2 * 1 / (h-1)
                # delete the points that are not in the original grid
                input_xyz_mat = input_xyz_mat[np.all(input_xyz_mat >= 0, axis=1) & np.all(input_xyz_mat <= 1, axis=1)]
                self.num_mats = len(input_xyz_mat)
                
                input_xyz = np.concatenate((input_xyz_geo, input_xyz_mat), axis=0)
                self.num_points = input_xyz.shape[0]
                
            # normalize the input_xyz
            for i in range(3):
                input_xyz[:, i] = normalize(input_xyz[:, i], 0, 1) # [-1, 1]
            
            input_d = normalize(np.power(np.power(input_xyz[:, 0], 2) + \
                np.power(input_xyz[:, 1], 2) + np.power(input_xyz[:, 2], 2), 0.5))
            input_b = np.ones(self.num_points)

            save_path = load_path
            save_content = np.concatenate((input_xyz, input_d.reshape(-1, 1), input_b.reshape(-1, 1)), axis=1)
            np.save(save_path, save_content)
        
        else:
            save_content = np.load(load_path)
            self.num_points = save_content.shape[0]
            if self.mode == "random":
                input_xyz = save_content[:, :3]
            else:
                input_xyz = save_content[:, :4]
            input_d = save_content[:, 3]
            input_b = save_content[:, 4]

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
        """Checks if the graph is a DAG, and returns accordingly.
        Return: 
            True if the graph is not a DAG (has cycles) False otherwise.
        """
        return not nx.is_directed_acyclic_graph(self.graph)

    def get_max_hidden_node_index(self):
        max_index = 0
        for input_node in nx.nodes(self.graph):
            if self.graph.nodes(data=True)[input_node]["type"] == "hidden" and int(input_node) >= max_index:
                max_index = input_node + 1
        return max_index

    def new_edge_is_valid(self, node1, node2):
        """ Checks that we are permitted to create an edge between these two nodes.
        New edges must:
        * not already exist
        * not create self loops (must be a feed forward network)
        Args:
            node1: Src node
            node2: Dest node
        Returns: 
            True if an edge can be added. False otherwise.
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