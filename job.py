import sys
import os
import random
import numpy
import subprocess as sub

from AFPO.afpomoo import AFPOMoo
from CPPN.Genome import Genotype, Phenotype
from CPPN.Networks import CPPN
from softbot_robot import SoftbotRobot
from utils import get_seq_num

import open3d as o3d
import numpy as np

# Genotype hyperparameters
IND_SIZE = (5, 5, 5)
DENSITY = 500.0
ONE_SHAPE = True
THRESHOLD = [0.5]
BORDER_WIDTH = 0.05
INSIDE_SPARSITY = 1

# AFPO hyperparameters
POP_SIZE = 8  # how large of a population are we using?
GENS = 10  # how many generations are we optimizing for?
NUM_FITS = 2 # how many fitness values are we maximizing?

if __name__ == '__main__':
    assert len(sys.argv) >= 2, "please run as python job.py seed"
    seed = int(sys.argv[1])

    numpy.random.seed(seed)
    random.seed(seed)

    # Setup evo run
    def robot_factory():
        genotype = Genotype(IND_SIZE, DENSITY, ONE_SHAPE, THRESHOLD, BORDER_WIDTH, INSIDE_SPARSITY)
        genotype.add_network(CPPN(output_node_names=["out"]))
        phenotype = Phenotype(genotype)
        return SoftbotRobot(phenotype, get_seq_num, seed, NUM_FITS)
    afpo_alg = AFPOMoo(robot_factory, pop_size=POP_SIZE)

    # do each generation.
    best_design = []
    for generation in range(GENS):
        print("generation %d" % (generation))
        dom_data = afpo_alg.generation()
        dom_inds = sorted(dom_data[1], key= lambda x: x.get_fitness(), reverse=False)
        print("%d individuals are dominating" % (dom_data[0],))
        print('\n'.join([str(d) for d in dom_inds]))

        best_fit, best_robot = afpo_alg.get_best()
        best_design = best_robot.morphology

    """
    Visualize best design
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(best_design[0][0][:, :3])
    colors = np.zeros([best_design[0][0].shape[0], 3])
    colors[:, :] = [1, 0, 0]
    cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([cloud])