from abc import ABCMeta, abstractmethod
import random

class Work(object):
    """
    Copied from ParallelPy to avoid the need to install ParallelPy for running
    """

    def cpus_requested(self):
        return 1

    def complete_work(self, serial=False):
        """
        Completes the required work, and generates a letter to send back to the dispatcher.
        :return: A letter to be sent.
        """
        self.compute_work(serial=serial)
        return self.write_letter()

    def compute_work(self, serial=False):
        """
        Entry point to do the required computation.
        :return: none
        """
        raise NotImplementedError

    def write_letter(self):
        """
        Generates a small packet of data, a Letter, to send back to the dispatcher for it to update itself with the
        completed work
        :return: A Letter to send back to the dispatcher to update the object with the new data
        :rtype: Letter
        """
        raise NotImplementedError

    def open_letter(self, letter):
        """
        A message to send to the dispatcher to update the Work with the resulting computation data.
        :param letter: the letter to open
        :return: None
        """
        raise NotImplementedError

class RobotInterface(Work):
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_id(self, new_id): 
        raise NotImplementedError

    @abstractmethod
    def get_id(self):
        """
        Return a unique identifer for this robot. Must be compareable to other robot ids.
        This is used as a tiebreaker in multi-objective optimization.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def iterate_generation(self):
        """
        This method will be called on each robot in the population every generation.
        If you implementing AFPO, then use this to update the age.
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def needs_evaluation(self):
        """
        :return: True if you need to be evaluted, false otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def mutate(self):
        """
        Make some mutations. You decide what mutations to make.
        :return: None
       """
        raise NotImplementedError

    @abstractmethod
    def dominates(self, other): 
        raise NotImplementedError

    @abstractmethod
    def dominates_final_selection(self, other): 
        raise NotImplementedError

    @abstractmethod
    def get_fitness(self): 
        raise NotImplementedError

class MOORobotInterface(RobotInterface):
    __metaclass__ = ABCMeta
    @abstractmethod
    def get_maximize_vals(self): 
        raise NotImplementedError

    @abstractmethod
    def get_minimize_vals(self): 
        raise NotImplementedError

    @abstractmethod
    def get_seq_num(self): 
        return self.get_id()

    def dominates(self, other):
        """
        returns True if self dominates other
        :param other: the other Student to compare self to.
        :return: True if self dominates other, False otherwise.
        tip: list comparision: a > b is True if any element of a is greater than the corresponding element of b.
        """
        self_min_traits = self.get_minimize_vals()
        self_max_traits = self.get_maximize_vals()

        other_min_traits = other.get_minimize_vals()
        other_max_traits = other.get_maximize_vals()

        # all min traits must be at least as small as corresponding min traits
        if list(filter(lambda x: x[0] > x[1], zip(self_min_traits, other_min_traits))):
            return False

        # all max traits must be at least as large as corresponding max traits
        if list(filter(lambda x: x[0] < x[1], zip(self_max_traits, other_max_traits))):
            return False

        # any min trait smaller than other min trait
        if list(filter(lambda x: x[0] < x[1], zip(self_min_traits, other_min_traits))):
            return True

        # any max trait larger than other max trait
        if list(filter(lambda x: x[0] > x[1], zip(self_max_traits, other_max_traits))):
            return True

        return self.get_seq_num() < other.get_seq_num()
