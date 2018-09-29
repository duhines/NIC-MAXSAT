import parse_input as parse
import sys
import random


class Parameters:
    def __init__(self, file_name, pop_size,
                 mutation_prob, mutation_shift,
                 num_generations,
                 ind_to_include, alpha):
        self.file_name = file_name
        self.pop_size = int(pop_size)
        self.mutation_prob = float(mutation_prob)
        self.mu_shift = mutation_shift
        self.num_generations = int(num_generations)
        self.ind_to_incl = int(ind_to_include)
        self.alpha = float(alpha)


# A class to look after our population vector:
class PopulationVector:
    def __init__(self, num_lits):
        # P = 0.5 for every literal to start with
        self.num_lits = num_lits
        self.vector = [0.5] * int(num_lits)

    # A method that allows you to generate a population from a givern popVector
    def generate_population(self, pop_size):
        population = []
        individual = []
        for i in range(pop_size):
            for j in range(self.num_lits):
                individual.append(random.random() > self.vector[j])
            population.append(individual.copy())
            del individual[:]
        return population

    # A method to update the probability vector based on the N best individuals
    def update_vector(self, scored_pop, ind_to_incl, alpha):
        for individual in scored_pop[0:ind_to_incl]:
            for i in range(self.num_lits):
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
        score = fitness(individual, problem)
        scored_generation.append((individual, score))
    return sorted(scored_generation, reverse=True)


def fitness(individual, problem):
    fitness = 0
    for clause in problem["clauses"]:
        check = False
        for literal in clause:
            if literal > 0:
                check = check or individual[literal - 1]
            else:
                check = check or not individual[abs(literal) - 1]
        if check:
            fitness += 1
    return fitness


def pbil(problem, parameters):
    pop_vector = PopulationVector(problem["num_literals"])

    # The following is the actual PBIL algorithm:
    iteration = 0
    while iteration < parameters.num_generations:
        nthPop = pop_vector.generate_population(parameters.pop_size)
        nthPop = score_pop(nthPop, problem)
        # Update pop vector using CSL approach described in paper:
        pop_vector.update_vector(nthPop,
                                 parameters.ind_to_include,
                                 parameters.alpha)
        pop_vector.mutate_vector(parameters.mutation_prob,
                                 parameters.mu_shift)

    # Final population vector (hopefully) will approximate correct solution
    final_pop = [round(x for x in pop_vector)]
    return(final_pop, pop_vector)