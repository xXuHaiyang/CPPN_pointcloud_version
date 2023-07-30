import uuid
import numpy as np
import pickle
import math
from PIL import Image

from AFPO import MOORobotInterface

class SoftbotRobot(MOORobotInterface):
    def __init__(self, phenotype, seq_num_gen, seed, num_fitnesses):
        self.seed = seed
        self.seq_num_gen = seq_num_gen
        self.seq_num = self.seq_num_gen()
        self.phenotype = phenotype
        self.morphology = self.phenotype.mophology
        assert self.morphology is not None, "Morphology should not be None!"

        self.id = self.set_uuid()
        
        self.fitnesses = [0 for x in range(num_fitnesses)]
        self.needs_eval = True

        self.age = 0
        self.generation = 0

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        repr_string = "AFPO BOT: fitness: "
        for fitness in self.get_fitness():
            repr_string += "%f, "%(fitness)
        repr_string += "age: %d, generation: %d, -- ID: %s" % (self.get_age(), self.get_generation(), str(self.seq_num))
        return repr_string

    def get_stats(self):
        stats_string = "%d,%d"%(self.get_id(), self.get_generation())
        for fitness in self.get_fitness():
            stats_string += ",%f"%(fitness)
        return stats_string + "\n"

    def get_id(self):
        return self.seq_num

    def set_id(self, new_id):
        self.seq_num = new_id
        
    def get_generation(self):
        return self.generation
    
    def set_generation(self, new_gen):
        self.generation = new_gen

    # Methods for MOORObotInterface class
    def iterate_generation(self):
        self.age += 1

    def needs_evaluation(self):
        return self.needs_eval

    def mutate(self):
        self.needs_eval = True
        self.phenotype.mutate()
        
        self.morphology = self.phenotype.mophology

        self.fitnesses = [0 for x in range(len(self.fitnesses))]
        self.age = 0

        self.set_uuid()
        self.seq_num = self.seq_num_gen()

    def get_minimize_vals(self):
        return [self.get_age()]

    def get_maximize_vals(self):
        return self.get_fitness()

    def get_seq_num(self):
        return self.seq_num

    def get_fitness(self, test=False):
        return self.fitnesses

    def dominates_final_selection(self, other):
        """
        Used for printing generation summary statistics -- we only print the pareto frontier.
        :param other: The other SoftbotRobot to compare with this one
        :return: True if robot dominates the other robot, false otherwise.
        """
        dominates = True
        for index in range(len(self.get_fitness())):
            dominates = dominates and (self.get_fitness()[index] > other.get_fitness()[index])
        return dominates

    # Methods for Work class
    def cpus_requested(self):
        return 1

    # want a ellipse with a = 1/3, b = 1/4, c = 1/5
    def ellipse_test_score(self, test_split):
        penalty = 0
        for point in test_split:
            if 9 * (point[0])**2 + 16 * (point[1])**2 + 25 * (point[2])**2 > 1:
                penalty -= 10
        return penalty
    
    # want a sphere with radius = 1/2
    def sphere_test_score(self, test_split):
        penalty = 0
        for point in test_split:
            if (point[0])**2 + (point[1])**2 + (point[2])**2 > 0.5:
                penalty -= 10
        return penalty

    # want a sphere with radius = 1/2 and a ellipse with a = 1/3, b = 1/4, c = 1/5
    def compute_work(self, test=True, **kwargs):
        cppn_morphology = self.morphology[0]
        
        self.fitnesses[0] = self.sphere_test_score(cppn_morphology[0])

    def write_letter(self):
        """
        When using ParallelPy with MPI for distributing simulations across multiple nodes,
        we only sync metadata about the simulation back to the evolutionary algorithm dispatch node.
        """
        return self.fitnesses

    def open_letter(self, letter):
        self.fitnesses = letter
        self.needs_eval = False
        return None

    def set_uuid(self):
        self.id = uuid.uuid1()
        return self.id

    def get_num_evaluations(self, test=False):
        return 1

    def get_age(self):
        return self.age

    def _flatten(self, l):
        ret = []
        for items in l:
            ret += items
        return ret

    def get_img(self):
        morphology = self.morphology[:, :, 0]
        imarray = (morphology * 255).astype(np.uint8)
        return Image.fromarray(imarray, mode='L')