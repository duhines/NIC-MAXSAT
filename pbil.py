import parse_input as parse
import genetic_alg as gCode
import sys
import random

# A class to look after our population vector:
class PopulationVector:
    def __init__(self, numLits):
        # P = 0.5 for every literal to start with
        self.numLits = numLits
        self.vector = [0.5] * int(numLits)

    # A method that allows you to generate a population from a givern popVector
    def generate_population(self, pop_size):
        population = []
        individual = []
        for i in range(0, pop_size):
            for j in range(0, self.numLits):
                individual.append(random.random() > self.vector[j])
            population.append(individual.copy())
            del individual[:]
        return population


    # A method to update the probability vector based on the N best individuals
    def update_vector(self, scored_pop, ind_to_incl, alpha):
        for individual in scored_pop[0:ind_to_incl]:
            for i in range(num_lits):
                if individual[i]:
                    self.vector[i] = self.vector[i](1 - alpha) + alpha
                else:
                    self.vector[i] = self.vector[i](1 - alpha)


    # Mutate pop vector. Mu := P(mutation). Shift := degree of mutation
    def mutate_vector(self, mu, shift):
        for prob in self.vector:
            if random.random() < mu:
                if random.random() < 0.5:
                    prob += shift
                else:
                    prob -= shift


def score_pop(population, problem):
    scored_generation = []
    for individual in population:
        score = gCode.fitness(individual, problem)
        scored_generation.append((individual, score))
    return sorted(scored_generation, reverse=True)

