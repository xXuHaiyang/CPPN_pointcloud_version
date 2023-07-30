import random
import numpy as np
from skimage.measure import marching_cubes
from copy import deepcopy

from .Networks import CPPN as Network
from .NetworkUtils import scale

# --------------------------------------------------------------------------------------------- #
# for test stability, we need to make sure the network is deterministic
# when realistcally used, we should eliminate the deterministic part
# random.seed(0)
# np.random.seed(0)
# --------------------------------------------------------------------------------------------- #

class Genotype(object):
    """A container for multiple networks, 'genetic code' copied with modification to produce offspring."""

    def __init__(self, orig_size_xyz=(5, 5, 5), density=10., one_shape_only=False, threshold=None, border_width=0.015, inside_sparsity=100):

        """
        Args:
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
        """Selects one network at random and attempts to mutate it."""
        random.choice(self.networks).mutate()

    def add_network(self, network):
        """Append a new network to this list of networks.
        Args:
            network : Network
                The network to add. Should be a subclass of Network.

        """
        assert isinstance(network, Network)
        self.networks += [network]
        self.all_networks_outputs.extend(network.output_node_names)

    def express(self, structural):
        """Calculate the genome networks outputs, the physical properties of each voxel for simulation"""
        for network in self.networks:
            for name in network.graph.nodes():
                network.graph.nodes[name]["evaluated"] = False  # flag all nodes as unevaluated

            network.set_input_node_states(self.orig_size_xyz, self.density)  # reset the inputs

            for name in network.output_node_names:
                network.graph.nodes[name]["state"] = np.zeros(network.num_points)  # clear old outputs
                network.graph.nodes[name]["state"] = self.calc_node_state(network, name)  # calculate new outputs
                network.graph.nodes[name]["state"] = scale(network.graph.nodes[name]["state"])  # scale outputs
                if structural:
                    network.morphology = geno_to_pheno_structual(network.pointcloud, \
                        network.num_geos, network.num_mats, network.graph.nodes[name]["state"], self.threshold, self.border_width, self.inside_sparsity)
                else:
                    network.morphology = geno_to_pheno(network.pointcloud, \
                        network.graph.nodes[name]["state"], self.threshold, self.border_width, self.inside_sparsity)

    def calc_node_state(self, network, node_name):
        """Propagate input values through the network"""
        if network.graph.nodes[node_name]["evaluated"]:
            return network.graph.nodes[node_name]["state"]

        network.graph.nodes[node_name]["evaluated"] = True
        input_edges = network.graph.in_edges(nbunch=[node_name])
        
        new_state = np.zeros(network.num_points)

        for edge in input_edges:
            node1, node2 = edge
            new_state += self.calc_node_state(network, node1) * network.graph.edges[node1, node2]["weight"]

        network.graph.nodes[node_name]["state"] = new_state

        return network.graph.nodes[node_name]["function"](new_state)

def geno_to_pheno(pointcloud, output_state, threshold, border_width=0.02, inside_sparsity=1000, add_cube_border=False):
    """Split the pcd according to the output space values
    Args:
        pcd: np.array
            pcd to split
        output_state: np.array
            output space of the pcd
        threshold: list[float, float, ...]
            threshold for the output space to split the pcd
        border_width: float, optional
            width of the border. Defaults to 0.01.
        inside_sparsity: float, optional
            sparsity of the inside points. Defaults to 100.
        add_cube_border: bool, optional
            whether to add a cube border. Defaults to False.
    Returns:
        split_parts: [part1(np.array), part2, part3...]
                    = [(x, y, z, output_state, split_index, geo_or_mat), 
                       (x, y, z, output_state, split_index, geo_or_mat),
                       (x, y, z, output_state, split_index, geo_or_mat)...]
            where: 
                split_index is the index of the split the point belongs to
                geo_or_mat is 0(geometry points) or 1(material points)
    """
    pcd = pointcloud
    N = pcd.shape[0]
    
    # --------------------------------------------------------------------------------------------- #
    # get threshold ready
    if type(threshold) != list:
        threshold = [threshold]
    if min(output_state) > max(threshold) or max(output_state) < min(threshold):
        return [np.random.rand(N, 6)-0.5] # TODO fix initial wrong, ugly imple
        # print("min(output_state):{}, max(output_state):{}, threshold:{}".format(min(output_state), max(output_state), threshold))
        # import sys
        # sys.exit("Error: output_state out of threshold range")
    threshold = [min(output_state)] + threshold + [max(output_state)]
    threshold.sort()
    # --------------------------------------------------------------------------------------------- #
    
    # --------------------------------------------------------------------------------------------- #
    # split the pcd
    split_parts = []
    
    for i in range(len(threshold)-1):
        part = None
        
        # --------------------------------------------------------------------------------------------- #
        # index the inside points and border points (cross-materials)
        if i == 0: # first split, no floor border
            if_inside = output_state < threshold[i+1]-border_width
            if_border = np.logical_and(threshold[i+1]-border_width <= output_state, output_state < threshold[i+1])
        elif i == len(threshold)-2: # last split, no ceiling border
            if_inside = output_state > threshold[i]+border_width
            if_border = np.logical_and(threshold[i] <= output_state, output_state <= threshold[i]+border_width)
        else:
            if_inside = np.logical_and(output_state > threshold[i]+border_width, output_state < threshold[i+1]-border_width)
            if_border = np.logical_and(threshold[i] <= output_state, output_state <= threshold[i]+border_width) \
                + np.logical_and(threshold[i+1]-border_width <= output_state, output_state < threshold[i+1])
        # --------------------------------------------------------------------------------------------- #

        # --------------------------------------------------------------------------------------------- #
        # arrange cross border and inside points
        border_part = pcd[if_border]
        inside_part = pcd[if_inside][::inside_sparsity]
        output_border_part = output_state[if_border]
        output_inside_part = output_state[if_inside][::inside_sparsity]
        split_border_part = np.ones(len(border_part), dtype=np.int8) * i
        split_inside_part = np.ones(len(inside_part), dtype=np.int8) * i
        # --------------------------------------------------------------------------------------------- #

        # --------------------------------------------------------------------------------------------- #
        # combine cross border and inside points to output
        # for border_part, all are geometry points, material_index is zero
        border_geo_or_mat = np.zeros(len(border_part))
        # fotmat: (x, y, z, output_state, split_index, geo_or_mat)
        border = np.concatenate((border_part, output_border_part.reshape(-1, 1), \
            split_border_part.reshape(-1, 1), border_geo_or_mat.reshape(-1, 1)), axis=1)
        # for inside_part, randomly 1/2 are material points(1), 1/2 are geometry points(0)
        inside_geo_or_mat = np.zeros(len(inside_part))
        inside_geo_or_mat[np.random.choice(len(inside_part), int(len(inside_part)/2), replace=False)] = 1
        # fotmat: (x, y, z, output_state, split_index, geo_or_mat)
        inside = np.concatenate((inside_part, output_inside_part.reshape(-1, 1), \
            split_inside_part.reshape(-1, 1), inside_geo_or_mat.reshape(-1, 1)), axis=1)
        part = np.concatenate((border, inside), axis=0)
        split_parts.append(part)
        # --------------------------------------------------------------------------------------------- #
    
    if add_cube_border:
        part = None
        # --------------------------------------------------------------------------------------------- #
        # default cube border, set on the 6 faces of the cube
        per_edge = 2*int(N**(1./3))
        s1 = np.linspace(-1, 1, per_edge, endpoint=True)
        s2 = np.linspace(-1, 1, per_edge, endpoint=True)
        s1, s2 = np.meshgrid(s1, s2)
        sp1 = np.ones_like(s1)
        sm1 = -sp1
        xp1 = np.stack((sp1, s1, s2), axis=-1).reshape(-1, 3)
        xm1 = np.stack((sm1, s1, s2), axis=-1).reshape(-1, 3)
        yp1 = np.stack((s1, sp1, s2), axis=-1).reshape(-1, 3)
        ym1 = np.stack((s1, sm1, s2), axis=-1).reshape(-1, 3)
        zp1 = np.stack((s1, s2, sp1), axis=-1).reshape(-1, 3)
        zm1 = np.stack((s1, s2, sm1), axis=-1).reshape(-1, 3)
        cube_part = np.concatenate((xp1, xm1, yp1, ym1, zp1, zm1), axis=0)
        cube_part = np.unique(cube_part, axis=0) # corner points are duplicated
        output_cube_part = np.ones(cube_part.shape[0]) * (-1)
        split_cube_part = np.ones(cube_part.shape[0], dtype=np.int8) * (-1)
        cube_geo_or_mat = np.zeros(cube_part.shape[0])
        # fotmat: (x, y, z, output_state, split_index, geo_or_mat)
        part = np.concatenate((cube_part, output_cube_part.reshape(-1, 1), \
            split_cube_part.reshape(-1, 1), cube_geo_or_mat.reshape(-1, 1)), axis=1)
        split_parts.append(part)
        # --------------------------------------------------------------------------------------------- #
        
    return split_parts

def geno_to_pheno_structual(pointcloud, num_geos, num_mats, output_state, threshold, border_width=0.02, inside_sparsity=1):
    """Split the structual pcd according to the output space values
    Args:
        pcd: np.array
            pcd to split
        output_state: np.array
            output space of the pcd
        threshold: list[float, float, ...]
            threshold for the output space to split the pcd
        border_width: float, optional
            width of the border. Defaults to 0.01.
        inside_sparsity: float, optional
            sparsity of the inside points. Defaults to 100.
    Returns:
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
    """
    # settings
    assert num_geos + num_mats == pointcloud.shape[0], "num_geos + num_mats != pointcloud.shape[0]"
    pcd = pointcloud
    N = num_geos
    l = int(N**(1./3)+0.5)
    assert l**3 == N, "pcd should be a cube"
    geo_or_mat = np.ones(num_geos + num_mats)
    geo_or_mat[:num_geos] = 0
    
    # --------------------------------------------------------------------------------------------- #
    # get threshold ready
    if type(threshold) != list:
        threshold = [threshold]
    if min(output_state) > max(threshold) or max(output_state) < min(threshold):
        return [np.random.rand(N, 6)-0.5] # TODO fix initial wrong, ugly imple
        # print("min(output_state):{}, max(output_state):{}, threshold:{}".format(min(output_state), max(output_state), threshold))
        # import sys
        # sys.exit("Error: output_state out of threshold range")
    threshold = [min(output_state)] + threshold + [max(output_state)]
    threshold.sort()
    # --------------------------------------------------------------------------------------------- #
    
    split_parts = {}
    inside_points = []
    border_points = []
    
    for i in range(len(threshold)-1):    
        part = None
        
        # --------------------------------------------------------------------------------------------- #
        # index the inside points
        if i == 0: # first split, no floor border
            if_inside = output_state < threshold[i+1]-border_width
        elif i == len(threshold)-2: # last split, no ceiling border
            if_inside = output_state > threshold[i]+border_width
        else:
            if_inside = np.logical_and(output_state > threshold[i]+border_width, output_state < threshold[i+1]-border_width)
        # --------------------------------------------------------------------------------------------- #

        # --------------------------------------------------------------------------------------------- #
        # arrange inside points
        inside_part = pcd[if_inside][::inside_sparsity]
        output_inside_part = output_state[if_inside][::inside_sparsity]
        split_inside_part = np.ones(len(inside_part), dtype=np.int8) * (i+1)
        inside_geo_or_mat = geo_or_mat[if_inside][::inside_sparsity]
        # --------------------------------------------------------------------------------------------- #

        # --------------------------------------------------------------------------------------------- #
        # send inside points to output
        # format: (x, y, z, output_state, split_index, geo_or_mat)
        inside = np.concatenate((inside_part[:, :3], output_inside_part.reshape(-1, 1), \
            split_inside_part.reshape(-1, 1), inside_geo_or_mat.reshape(-1, 1)), axis=1)
        inside_points.append(inside)
        # --------------------------------------------------------------------------------------------- #
        
    # --------------------------------------------------------------------------------------------- #
    # marching cube to get the border points
    pcd_for_marching_cube = np.zeros((l, l, l))
    pcd_for_fill = ((pcd[:num_geos] + 1) / 2 * (l-1)).astype(int)
    for i in range(N):
        pcd_for_marching_cube[pcd_for_fill[i, 0], pcd_for_fill[i, 1], pcd_for_fill[i, 2]] = output_state[i]
    # --------------------------------------------------------------------------------------------- #
    
    # --------------------------------------------------------------------------------------------- #
    # send border points to output
    from skimage import measure
    for i in range(1, len(threshold)-1):
        verts, faces, normals, values = measure.marching_cubes(pcd_for_marching_cube, threshold[i], allow_degenerate=False)
        verts = verts / (l-1) * 2 - 1
        border = (verts, faces, normals, values)
        border_points.append(border)
    # --------------------------------------------------------------------------------------------- #
    
    split_parts["inside"] = inside_points
    split_parts["border"] = border_points

    return split_parts

class Phenotype(object):
    """Physical manifestation of the genotype - determines the physiology of an individual."""

    def __init__(self, genotype, structural=False):

        """
        Args:
            genotype : Genotype cls
                Defines particular networks (the genome).
            structural : bool
                Defines whether the phenotype is structural or not
        """
        self.structural = structural
        
        repeat = 0
        self.genotype = genotype
        self.genotype.express(structural=self.structural)

        while not self.is_valid():
            self.genotype = genotype()
            self.genotype.express(structural=self.structural)
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
            self.genotype.express(self.structural)
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
