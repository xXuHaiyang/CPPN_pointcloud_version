import random
from copy import deepcopy
from scipy.ndimage.measurements import label

import numpy as np

from .NetworkUtils import sigmoid
from .Networks import CPPN as Network

# # for test stability, we need to make sure the network is deterministic
# # when realistcally used, we should eliminate the deterministic part
# random.seed(0)
# np.random.seed(0)

class Genotype(object):
    """A container for multiple networks, 'genetic code' copied with modification to produce offspring."""

    def __init__(self, orig_size_xyz=(6, 6, 6), density=10., one_shape_only=False, threshold=None, border_width=0.05, inside_sparsity=1000):

        """
        Parameters
        ----------
        orig_size_xyz : 3-tuple (x, y, z)
            Defines the original 3 dimensions for the cube of voxels corresponding to possible networks outputs. The
            maximum number of SofBot voxel components is x*y*z, a full cube.
        density: float x
            Defines the density of fill in pointclouds
        one_shape_only: bool
            Defines whether the genotype can only express one shape
        threshold: float x / list
            Defines the threshold for the pointclouds materials thesholds
        border_width: float x
            Defines the border width between materials
        inside_sparsity: float x
            Defines the sparsity of the inside of the pointclouds
        """
        self.networks = []
        self.all_networks_outputs = []
        self.orig_size_xyz = orig_size_xyz
        self.density = density
        self.one_shape_only = one_shape_only
        self.threshold = threshold
        self.border_width = border_width
        self.inside_sparsity = inside_sparsity
        
        self.flatten_size = int(self.orig_size_xyz[0] * self.orig_size_xyz[1] * self.orig_size_xyz[2] * self.density)

    def __iter__(self):
        """Iterate over the networks. Use the expression 'for n in network'."""
        return iter(self.networks)

    def __len__(self):
        """Return the number of networks in the genotype. Use the expression 'len(network)'."""
        return len(self.networks)

    def __getitem__(self, n):
        """Return network n.  Use the expression 'network[n]'."""
        return self.networks[n]

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def __str__(self):
        return "Genotype made of: " + ',\n'.join(map(str, self.networks))

    def mutate(self, **kwargs):
        """
        Selects one network at random and attempts to mutate it.
        :param kwargs: Not currently used.
        :return: None
        """
        random.choice(self.networks).mutate()

    def add_network(self, network):
        """Append a new network to this list of networks.

        Parameters
        ----------
        network : Network
            The network to add. Should be a subclass of Network.

        """
        assert isinstance(network, Network)
        self.networks += [network]
        self.all_networks_outputs.extend(network.output_node_names)

    def express(self):
        """Calculate the genome networks outputs, the physical properties of each voxel for simulation"""
        for network in self.networks:
            for name in network.graph.nodes():
                network.graph.nodes[name]["evaluated"] = False  # flag all nodes as unevaluated

            network.set_input_node_states(self.orig_size_xyz, self.density)  # reset the inputs
            flattened_output = int(self.orig_size_xyz[0] * self.orig_size_xyz[1] * self.orig_size_xyz[2] * self.density)

            for name in network.output_node_names:
                network.graph.nodes[name]["state"] = np.zeros(flattened_output)  # clear old outputs
                network.graph.nodes[name]["state"] = self.calc_node_state(network, name)  # calculate new outputs
                network.morphology = geno_to_pheno(network.pointcloud, \
                    network.graph.nodes[name]["state"], self.threshold, self.border_width, self.inside_sparsity)

    def calc_node_state(self, network, node_name):
        """Propagate input values through the network"""
        if network.graph.nodes[node_name]["evaluated"]:
            return network.graph.nodes[node_name]["state"]

        network.graph.nodes[node_name]["evaluated"] = True
        input_edges = network.graph.in_edges(nbunch=[node_name])
        
        new_state = np.zeros(self.flatten_size)

        for edge in input_edges:
            node1, node2 = edge
            new_state += self.calc_node_state(network, node1) * network.graph.edges[node1, node2]["weight"]

        network.graph.nodes[node_name]["state"] = new_state

        return network.graph.nodes[node_name]["function"](new_state)

### INP
# NODE(geo): num
# id, x y z, volume
# MPT(material): num
# id, x y z, volume
# MPT_GROUP, category
# id, cat_id

def format_outprint(split_parts, filename):
    """format the split parts to the output format

    Args:
        split_parts (list): list of split parts
        filename (str): filename of the output file

    Returns:
        None
    """
    geo_num = 0
    mat_num = 0
    


def geno_to_pheno(pointcloud, output_state, threshold, border_width=0.05, inside_sparsity=1000):
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
        geo_or_mat is 0(geometry points) or 1(material points)
    """
    if type(threshold) != list:
        threshold = [threshold]
    # TODO
    if min(output_state) > max(threshold) or max(output_state) < min(threshold):
        return [np.ones((10, 6))]
    threshold = [min(output_state)] + threshold + [max(output_state)]
    threshold.sort()
    
    split_parts = []
    for i in range(len(threshold)-1):
        part = None
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


class Phenotype(object):
    """Physical manifestation of the genotype - determines the physiology of an individual."""

    def __init__(self, genotype):

        """
        Parameters
        ----------
        genotype : Genotype cls
            Defines particular networks (the genome).

        """
        repeat = 0
        self.genotype = genotype
        self.genotype.express()

        while not self.is_valid():
            self.genotype = genotype()
            self.genotype.express()
            print("From Phenotype: Invalid phenotype, regenerating {}...".format(repeat))
            repeat += 1
            
        self.mophology = []
        for network in self.genotype.networks:
            self.mophology.append(network.morphology)

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def mutate(self, max_mutation_attempts=1000):
        """
        Mutates the genotype and ensures that the mutation results in a valid phenotype.
        TODO: Do we want to ensure that the mutated genome is actually different from the main genome?
        TODO: ^yes
        :param max_mutation_attempts: Try to find a valid mutation up to this many times.
        :return: True if mutation occured, false otherwise.
        """
        old_genotype = self.genotype
        for i in range(max_mutation_attempts):
            self.genotype = deepcopy(old_genotype)
            self.genotype.mutate(max_mutation_attempts=max_mutation_attempts)
            self.genotype.express()
            if self.is_valid():
                return True
        return False

    def is_valid(self):
        """Ensures a randomly generated phenotype is valid (checked before adding individual to a population).

        Returns
        -------
        is_valid : bool
        True if self is valid, False otherwise.

        """
        for network in self.genotype:
            for output_node_name in network.output_node_names:
                if np.isnan(network.graph.nodes[output_node_name]["state"]).any():
                    return False

        return True
