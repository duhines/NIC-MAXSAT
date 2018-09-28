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
MAXSAT_PROBLEM = []


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
    def __init__(self, generation, pop_size):
        self.generation = generation
        self.pop_size = pop_size

    def next_generation(self):
        self.generation = self.generation + 1

    def generate_initial_pop(self, problem):
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
        solution = []
        for i in range(0, self.pop_size):  # Make N individuals
            for j in range(0, problem["num_literals"]):  # Make indivs. proper len
                solution.append(random.random() > .5)
            new_individual = Individual(solution.copy())
            population.append(new_individual)  # Copy so we don't lose reference

            del solution[:]

        self.individuals = population

    def score_individuals(self):
        for individual in self.individuals:
            individual.get_fitness(MAXSAT_PROBLEM) 

    def select(self, selection_method):
        """
        Purpose: Wrapper method for the selection methods.  Used to call the correct
            selection method based on the command line argument specifying the selection
            method.
        Parameters:
        """
        if selection_method == "rs":
            self.rank_select()
        elif selection_method == "ts":
            self.tournament_select()
        elif selection_method == "bs":
            self.boltzmann_select()

    def recombination(self, xover_prob, xover_type):
        """
        Purpose: From the selected individuals, crossover by chance xover_prob to create
            next generation.
        Parameters: The selected individuals from the current generation, the probability
            of performing crossover, and the crossover type.
        Return: The next generation of individuals!
        """
        next_gen = []
        for i in range(0, len(self.individuals)//2):
            first = self.individuals[random.randint(0, len(self.individuals) - 1)]
            second = self.individuals[random.randint(0, len(self.individuals) - 1)]
            if random.random() < xover_prob:
                if xover_type == "1c":
                    offspring = self.single_crossover(first, second)
                else:
                    offspring = self.uniform_crossover(first, second)
                next_gen.append(offspring[0])
                next_gen.append(offspring[1])
            else:
                next_gen.append(first)
                next_gen.append(second)
        del self.individuals[:]
        self.individuals = next_gen.copy()

    def mutate(self, mutation_prob):
        """
        Purpose: Mutate the population!
        Parameters: Population represented as a list of individuals where
            each individual is a list of booleans and a probability for
            mutation.
        Return: The population, but mutated!
        """
        for individual in self.individuals:
            for literal in individual.solution:
                if random.random() < mutation_prob:
                    literal = not literal

    def rank_select(self):
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
        sorted_generation = sorted(self.individuals, key=lambda individual: individual.fitness)
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
            selected = sorted_generation[increment - 1]
            selected_individuals.append(selected)

        del self.individuals[:]
        self.individuals = selected_individuals

    def tournament_select(self):
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
        for i in range (0, len(self.individuals)):
            rand_index1 = random.randint(0, len(self.individuals) - 1)
            rand_index2 = random.randint(0, len(self.individuals) - 1)
            individual1 = self.individuals[rand_index1]
            individual2 = self.individuals[rand_index2]
            #compare individual fitnesses
            if individual1.fitness > individual2.fitness:
                selected_individuals.append(individual1)
            else:
                selected_individuals.append(individual2)
        
        del self.individuals[:]
        self.individuals = selected_individuals

    def boltzmann_select(self):
        return

    def single_crossover(self, individual1, individual2): 
        """
        Purpose: From two individuals, use a single crossover point to produce two children.
        Parameters: Two individuals, each representing a MAXSAT solution as a list of Boolean
            values.
        Return: A tuple of the two two children produced by the single point crossover.  
        """
        crossover_point = random.randint(1, len(individual1.solution) - 2)
        first_child_solution = individual1.solution[:crossover_point].copy() + individual2.solution[crossover_point:].copy()
        second_child_solution = individual2.solution[:crossover_point].copy() + individual1.solution[crossover_point:].copy()
        first_child = Individual(first_child_solution)
        second_child = Individual(second_child_solution)
        first_child.get_fitness(MAXSAT_PROBLEM)
        second_child.get_fitness(MAXSAT_PROBLEM)
        return (first_child, second_child)


    def uniform_crossover(self, individual1, individual2):
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

        for index in range(0, len(breeding_pair.solution)):
            flip_for_first = random.random()
            flip_for_second = random.random()
            if flip_for_first < .5:
                first_child.append(breeding_pair[0].solution[index])
            else:
                first_child.append(breeding_pair[1].solution[index])
            if flip_for_second < .5:
                worse_child.append(breeding_pair[0].solution[index])
            else:
                worse_child.append(breeding_pair[1].solution[index])

        return (first_child, worse_child)


class Individual:
    def __init__(self, bools_array):
        self.solution = bools_array

    def get_fitness(self, problem):
        """
        Purpose: For a given individual and the clauses for the problem, evaluate
            the number of correct clauses and return this number as the fitness.
        Input: An individual representing a potential solution as a list of
            Boolean values.
        Return: An integer indicating the number of clauses that the individual's
            solution made true.
        """
        fitness = 0
        for clause in problem["clauses"]:
            check = False
            for literal in clause:
                if literal > 0:
                    check = check or self.solution[literal - 1]
                else:
                    check = check or not self.solution[abs(literal) - 1]
            if check:
                fitness += 1
        self.fitness = fitness
        return fitness


class BestSoFar:
    def __init__(self, individual, iteration):
        self.individual = individual
        self.iteration_found = iteration

    def compare_to_best(self, individuals, iteration):
        for individual in individuals:
            if individual.fitness > self.individual.fitness:
                self.individual.solution = individual.solution.copy()
                self.individual.fitness = individual.fitness
                self.iteration_found = iteration
                print("Found new best with score {} in generation {}".format(self.individual.fitness, self.iteration_found))
                return True

        return False


def pretty_solution(solution):
    pretty = ""
    ith_literal = 1
    ten_per_line = 0
    for literal in solution:
        pretty = pretty + "L" + str(ith_literal) + ": " + str(literal) + "  "
        ith_literal = ith_literal + 1
        ten_per_line = ten_per_line + 1
        if ten_per_line > 10:
            ten_per_line = 0
            pretty = pretty + "\n"
    return pretty


def print_solution(best_so_far, problem, parameters):
    print("File: {}".format(parameters.file_name))
    print("Literals count: {}\nClauses count: {}".format(MAXSAT_PROBLEM["num_literals"], MAXSAT_PROBLEM["num_clauses"]))
    percentage_correct = round((best_so_far.individual.fitness / MAXSAT_PROBLEM["num_clauses"]) * 100, 1)
    print("Best individual scored {} ({}%)".format(best_so_far.individual.fitness, percentage_correct))
    print("Solution:\n{}".format(pretty_solution(best_so_far.individual.solution)))
    print("Found in iteraion {}".format(best_so_far.iteration_found))


def standard_GA(problem, parameters):
    """
    Purpose:
    Parameters:
    Return:
    """
    print(MAXSAT_PROBLEM)
    population = Population(0, parameters.pop_size)
    population.generate_initial_pop(MAXSAT_PROBLEM)
    population.individuals[0].get_fitness(MAXSAT_PROBLEM)
    best_so_far = BestSoFar(population.individuals[0], 0)
    if best_so_far.individual.fitness == MAXSAT_PROBLEM["num_clauses"]:
        print("Full Solution!")
        print_solution(best_so_far, problem, parameters)
        return
    iteration = 0
    while iteration < parameters.num_generations:
        print("beginning")
        for individual in population.individuals:
            print(individual.solution)
        population.next_generation()
        population.score_individuals()
        print("after score")
        for individual in population.individuals:
            print(individual.solution)
        #print(population.individuals)
        if best_so_far.compare_to_best(population.individuals, iteration):
            if best_so_far.individual.fitness == MAXSAT_PROBLEM["num_clauses"]:
                print("Full Solution!")
                print_solution(best_so_far, problem, parameters)
                return
        population.select(parameters.selection_type)
        print("after select")
        for individual in population.individuals:
            print(individual.solution)
        #print(population.individuals)
        population.recombination(parameters.xover_prob, parameters.xover_method)
        print("after recombination")
        for individual in population.individuals:
            print(individual.solution)
        #print(population.individuals)
        population.mutate(parameters.mutation_prob)
        print("after mutate")
        for individual in population.individuals:
            print(individual.solution)
        #print(population.individuals)
        iteration = iteration + 1
        print("Generation: {}".format(iteration))
        
    print_solution(best_so_far, problem, parameters)


def main():
    # acquire command line arguments
    global MAXSAT_PROBLEM
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
    MAXSAT_PROBLEM = sample_problem
    #MAXSAT_PROBLEM = parse.return_problem(FILE + parameters.file_name)
    solution = standard_GA(problem, parameters)


main()
