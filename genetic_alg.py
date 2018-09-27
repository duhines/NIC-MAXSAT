"""
1. Get the parser going.
    i. read cnf FILE->determine number of literals for the size of the individuals representation
    ii. put the FILE in an array, where each item in array is one clause from the FILE (one line)
        a. clause ex:  ..., 8, 3, -4,... (negative values indicate that the literal is negated)
    iii. save the boolean values as strings


Command Line arguments:
    python3 genetic_alg.py <file name> <population size> <selection method> <crossover method>
        <crossover probability> <mutation probability> <number of generations> <use GA/PBIL>

"""
"""
    STYLE: 
        - 80 chars max per line
        - lets use underscores instead of camel case
        - function docstrings inside functions
        - module docstring at top of each FILE with 
            - authors, purpose, bugs, etc.
"""

import parse_input as parse
import sys
import random

FILE = "problems/"

class Parameters:
    def __init__(self, file_name, pop_size, selection_type, xover_method, xover_prob, mutation_prob, num_generations, algorithm):
        self.file_name = file_name
        self.pop_size = int(pop_size)
        self.selection_type = selection_type 
        self.xover_method = xover_method
        self.xover_prob = float(xover_prob)
        self.mutation_prob = float(mutation_prob)
        self.num_generations = int(num_generations)
        self.algorithm = algorithm


class Population:
    def __init__(self, generation, individuals, size):
        self.generation = generation
        self.individuals = individuals
        self.size = size


class Individual:
    def __init__(self, bools_array, fitness):
        self.solution = bools_array
        self.fitness = fitness



class BestSoFar:
    def __init__(self, individual, iteration):
        self.individual = individual
        self.iteration_found = iteration

# Variable to monitor the "Global Best" solution that will be returned:
G_BEST = ([], 0)
def generate_initial_pop(problem, pop_size):
    """
    Purpose: Generate an initial population to start the genetic algorithm with.
    Input:
        problem: Dictionary including the following information: number of
            literals, number of clauses, and the clauses.
        population_size: integer indicating the number of individuals that
            are needed for the initial generation.
    Return: A list of the individuals in the population where each individual
        is a list of randomly assigned Boolean values.
    """
    population = []
    individual = []
    for i in range(0, pop_size):  # Make N individuals
        for j in range(0, problem["num_literals"]):  # Make indivs. proper len
            individual.append(random.random() > .5)
        population.append(individual.copy())  # Copy so we don't lose reference
        del individual[:]

    return population



def rank_select(scored_generation):
    """
    Purpose: Use rank selection to select individuals to progress to the recombination
        phase of the genetic algorithm.  In rank selection, individuals are sorted by
        fitness and then assigned a rank (1st best, 2nd best, etc.).  Individuals are
        then choosen randomly proportional to their rank such that higher ranked
        individuals are more likely to be chosen.
    Input: The current generation as a list of tuples where the first element of the
        tuple is the individual and the second element of the tuple is the individual's
        score.
    Return: List of individuals that were selected by ranked selection.
    """
    # via: https://stackoverflow.com/questions/3121979/how-to-sort-list-tuple-of-lists-tuples
    sorted_generation = sorted(scored_generation, key=lambda individual: individual[1])
    total_rank = 0
    for i in range(1, len(sorted_generation) + 1):
        total_rank += i

    selected_individuals = []
    for i in range(0, len(sorted_generation)):
        base = 0
        increment = 1
        rand_value = random.randint(0, total_rank - 1)
        while rand_value > base:
            increment += 1
            base += increment
        selected = sorted_generation[increment - 1][0]
        selected_individuals.append(selected)

    return selected_individuals


def fitness(individual, problem):
    """
    Purpose: For a given individual and the clauses for the problem, evaluate
        the number of correct clauses and return this number as the fitness.
    Input: An individual representing a potential solution as a list of
        Boolean values.
    Return: An integer indicating the number of clauses that the individual's
        solution made true.
    """
    global G_BEST
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
    if fitness > G_BEST[1]:
        G_BEST = (individual, fitness)
    return fitness


def tournament_select(scored_generation):
    """
    Purpose: Use tournament selection to determine which individuals progress
        to the recombination phase.  Tournament selection will repeatedly choose
        pairs from the population and then take the fitter of the two is selected.
    Input: The current generation as a list of tuples where the first element of the
        tuple is the individual and the second element of the tuple is the individual's
        score.
    Return: List of individuals that were selected by tournament selection.
    """
    selected_individuals = []
    for i in range (0, len(scored_generation)):
        rand_index1 = random.randint(0, len(scored_generation) - 1)
        rand_index2 = random.randint(0, len(scored_generation) - 1)
        individual1 = scored_generation[rand_index1]
        individual2 = scored_generation[rand_index2]
        #compare individual fitnesses
        if individual1[1] > individual2[1]:
            selected_individuals.append(individual1[0])
        else:
            selected_individuals.append(individual2[0])

    return selected_individuals


def boltzmann_select():
    return


def select(scored_generation, selection_method):
    """
    Purpose: Wrapper method for the selection methods.  Used to call the correct
        selection method based on the command line argument specifying the selection
        method.
    Parameters:
    """
    if selection_method == "rs":
        return rank_select(scored_generation)
    elif selection_method == "ts":
        return tournament_select(scored_generation)
    elif selection_method == "bs":
        return boltzmann_select(scored_generation)


def recombination(selected, xover_prob, xover_type):
    """
    Purpose: From the selected individuals, crossover by chance xover_prob to create
        next generation.
    Parameters: The selected individuals from the current generation, the probability
        of performing crossover, and the crossover type.
    Return: The next generation of individuals!
    """
    global G_BEST
    next_gen = []
    for i in range(0, len(selected)//2):
        first = selected[random.randint(0, len(selected) - 1)]
        second = selected[random.randint(0, len(selected) - 1)]
        if random.random() < xover_prob:
            if xover_type == "1c":
                offspring = single_crossover(first, second)
            else:
                #what dooooooo??  <<--------------------------------------------<<<|
                offspring = uniform_crossover(first, second)
            next_gen.append(offspring[0])
            next_gen.append(offspring[1])
        else:
            next_gen.append(first)
            next_gen.append(second)
    return next_gen


def single_crossover(individual1, individual2):	
    """
    Purpose: From two individuals, use a single crossover point to produce two children.
    Parameters: Two individuals, each representing a MAXSAT solution as a list of Boolean
        values.
    Return: A tuple of the two two children produced by the single point crossover.  
    """
    crossover_point = random.randint(1, len(individual1) - 2)
    first_child = individual1[:crossover_point].copy() + individual2[crossover_point:].copy()
    second_child = individual2[:crossover_point].copy() + individual1[crossover_point:].copy()
    return (first_child, second_child)


def uniform_crossover(individual1, individual2):
    """
    Purpose: Perform uniform crossover on two indiduals; each pair produces two offspring
        by flipping a coin to determine which parent passes down the literal assignment
        to the child
    Parameters: The two individuals that are making children
    Return: A tuple of the first and second children of individual1/2
        Note:
            - everyone has two children
            - the first child is the best of the bunch
    """
    breeding_pair = (individual1, individual2)
    first_child = []
    worse_child = [] #@dgans, @djanderson

    for index in range(0, len(breeding_pair[0])):
        flip_for_first = random.random()
        flip_for_second = random.random()
        if flip_for_first < .5:
            first_child.append(breeding_pair[0][index])
        else:
            first_child.append(breeding_pair[1][index])
        if flip_for_second < .5:
            worse_child.append(breeding_pair[0][index])
        else:
            worse_child.append(breeding_pair[1][index])

    return (first_child, worse_child)

def mutate(population, mutation_prob):
    """
    Purpose: Mutate the population!
    Parameters: Population represented as a list of individuals where
        each individual is a list of booleans and a probability for
        mutation.
    Return: The population, but mutated!
    """
    for individual in population:
        for literal in individual:
            if random.random() < mutation_prob:
                literal = not literal
    return population


def standard_GA(problem, parameters):
    """
    Purpose:
    Parameters:
    Return:
    """
    initial_generation = generate_initial_pop(problem, parameters.pop_size)

    iteration = 0
    current_generation = initial_generation.copy()
    while iteration < parameters.num_generations:
        iteration += 1
        scored_generation = []
        for individual in current_generation:
            score = fitness(individual, problem)
            scored_generation.append((individual, score))
        selected = select(scored_generation, parameters.selection_type)
        recomb_generation = recombination(selected, parameters.xover_prob, parameters.xover_method)
        if G_BEST[1] == problem["num_clauses"]:
            print("Solution found after {0} generations.".format(iteration))
            break
        mutated_generation = mutate(recomb_generation, parameters.mutation_prob)
        del current_generation[:]
        current_generation = mutated_generation.copy()
        print("Generation: {}".format(iteration))

    print("\n{0}\nIs best solution with score: {1}".format(G_BEST[0], G_BEST[1]))
    return


def main():
    # acquire command line arguments
    parameters = Parameters(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8])            

    if parameters.pop_size % 2 != 0:
        parameters.pop_size += 1
    """
    **For testing purposes only**
    """
    problem = [[1, -4], [-2, -3], [4, 1], [-4, 4], [-3, 1], [-1, 2], [1, 1], [-1, 1]]
    sample_problem = {
        "num_literals": 4,
        "num_clauses": 8,
        "clauses": problem
    }

    # Acquire MAXSAT problem
    problem = parse.return_problem(FILE + parameters.file_name)

    solution = standard_GA(problem, parameters)


main()
