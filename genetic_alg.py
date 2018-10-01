"""
Team: 3D^3
Authors: Dustin Hines, David Anderson, Duncan Gans
Course: Nature-Inspired Computation
Assignment: Evolutionary Algorithms for MAXSAT: A Comparison of Genetic Algorithms 
    and Population Based Incremental Learning
Date: 1 October 2018
Description:
    This file implements a genetic algorithm for solving MAXSAT problems.  Our 
    implementation includes:
        3 selection methods:
            - rank select
            - tournament select
            - Boltzmann section
        And 2 crossover methods:
            - single point crossover
            - uniform crossover 
    
    The genetic algorithm first creates a population of individuals with random 
    solutions to the MAXSAT problem.  Then, it iteratively selects individuals
    and recombines them with crossover with a certain probability.  After 
    recombination, each individual is mutated based on a another probability.
    The individuals in the population at this stage then progress to the next
    iteration of the algorithm, repeating this process.  After the algorithm 
    has gone through iterations equal to the specified number of generations,
    the best solution found is #printed along with the generation it was found
    in, the percentage of the clauses it satisfied, the name of the MAXSAT
    problem file, the solution of literal values, and the number of variables
    and clauses in the problem.

Notes:
    - To run:
        If using GA:
        python3 genetic_alg.py <file name> <population size> <selection method>
        <crossover method> <crossover probability> <mutation probability>
        <number of generations> <ga>

        If using PBIL:
        python3 genetic_alg.py <file_name> <population_size> <number_of_individuals
        for_CSL> <alpha> <mutation_step> <mutation_prob> <num_generations> <pbil>

        Example command line:
        python3 genetic_alg.py maxcut-140-630-0.7-1.cnf 100 1 0.1 0.01 0.01 1000 pbil

    - Our file assumes that the MAXSAT file name specified is in the folder
        "problems"
    - To import the MAXSAT problems, the parse_input module is imported.
    - PBIL code is located in pbil.py, which is imported. This file
        implements GA and also runs the program.
"""

# Required libraries
import parse_input as parse
import pbil as PBIL
import sys
import random
import math
# Used for testing: import time

FILE = "problems/"
MAXSAT_PROBLEM = []


class Parameters:
    """
    Purpose: Class used to store the command line arguments conveniently.
    This class is specific to the GA implementation; PBIL calls
    pbil.PBILParameters, as it has slightly different parameters.
    """
    def __init__(self, file_name, pop_size, selection_type, xover_method,
        xover_prob, mutation_prob, num_generations, algorithm):
        self.file_name = file_name
        self.pop_size = int(pop_size)
        self.selection_type = selection_type 
        self.xover_method = xover_method
        self.xover_prob = float(xover_prob)
        self.mutation_prob = float(mutation_prob)
        self.num_generations = int(num_generations)
        self.algorithm = algorithm


class Population:
    """
    Population for the GA.  Includes the following methods:
        - next_generation: 
        - generate_initial_pop
        - score_individuals
        - select
        - rank_select
        - tournament_select
        - boltzmann_select
        - recombination
        - single_crossover
        - uniform_crossover
        - mutate
    """
    def __init__(self, generation, pop_size):
        self.generation = generation
        self.pop_size = pop_size
        self.individuals = []

    def next_generation(self):
        """
        Purpose: increment the generation count by 1
        """
        self.generation = self.generation + 1

    def generate_initial_pop(self, num_literals):
        """
        Purpose: Generate an initial population to start the genetic algorithm
            and set self.individuals to this population.
        Input: number of literals
        Return: none
        """
        population = []
        solution = []
        for i in range(0, self.pop_size):  # Make N individuals
            for j in range(0, num_literals):  # Make indivs. proper len
                solution.append(random.random() > .5)
            new_individual = Individual(solution.copy())
            population.append(new_individual)  # Copy so we don't lose reference

            del solution[:]

        self.individuals = population

    def score_individuals(self, best_so_far):
        """
        Purpose: For each of the individuals in the population, if necessary,
            calculate their fitness and compare to the best individual so far
            to update that variable if a better solution is found/
        Input: best_so_far class
        Return: none
        """
        for individual in self.individuals:
            if individual.fitness == -1:
                individual.get_fitness(MAXSAT_PROBLEM)
                best_so_far.compare_to_best(individual, self.generation)


    def select(self, selection_method):
        """
        Purpose: Wrapper method for the selection methods. Used to call the correct
            selection method based on the command line argument specifying the selection
            method.
        Parameters: Selection method to use
        Return: none
        """
        if selection_method == "rs":
            self.rank_select()
        elif selection_method == "ts":
            self.tournament_select()
        elif selection_method == "bs":
            self.boltzmann_select()

    def recombination(self, xover_prob, xover_type):
        """
        Purpose: for individual in the population, crossover by chance
            xover_prob to create next generation.
        Parameters: the probability of performing crossover, and the crossover type.
        Return: none
        """
        next_gen = []
        for i in range(0, len(self.individuals)//2):
            first = self.individuals[i]
            second = self.individuals[i+len(self.individuals)//2]
            #recombine with xover_prob
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
        #clear individuals first
        del self.individuals[:]
        self.individuals = next_gen.copy()

    def mutate(self, mutation_prob):
        """
        Purpose: Mutate the individuals population by flipping a coin to decide
            whether to switch the value of each literal in the solution.
        Parameters: probability for mutation
        Return: none
        """
        for index, individual in enumerate(self.individuals):
            for index2, literal in enumerate(self.individuals[index].solution):
                # Mutate with probability specified
                if random.random() < mutation_prob:
                    self.individuals[index].solution[index2] = not literal
                    individual.fitness = -1

    def rank_select(self):
        """
        Purpose: Use rank selection to select individuals to progress to the recombination
            phase of the genetic algorithm.  In rank selection, individuals are sorted by
            fitness and then assigned a rank (1st best, 2nd best, etc.).  Individuals are
            then choosen randomly proportional to their rank such that higher ranked
            individuals are more likely to be chosen.
        Input: none needed since this is a method for population
        Return: none
        """
        # via:https://stackoverflow.com/questions/3121979/
        #    how-to-sort-list-tuple-of-lists-tuples
        sorted_generation = sorted(self.individuals, 
            key=lambda individual:individual.fitness)
        total_rank = 0
        for i in range(1, len(sorted_generation) + 1):
            total_rank += i

        selected_individuals = []
        for i in range(0, len(sorted_generation)):
            base = 0
            increment = 1
            rand_value = random.randint(0, total_rank - 1)
            #this is essentially implementing a roulette table
            #not sure how else to explain this 
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
        Input: none 
        Return: none
        """
        selected_individuals = []
        for i in range (0, len(self.individuals)):
            rand_index1 = random.randint(0, len(self.individuals) - 1)
            rand_index2 = random.randint(0, len(self.individuals) - 1)
            individual1 = self.individuals[rand_index1]
            individual2 = self.individuals[rand_index2]
            #compare individual fitnesses, choose best
            if individual1.fitness > individual2.fitness:
                selected_individuals.append(individual1)
            else:
                selected_individuals.append(individual2)
        
        del self.individuals[:]
        self.individuals = selected_individuals

    def boltzmann_select(self):
        """
        Purpose: Use Boltzmann selection to determine which individuals progress
            to the reconbination phase. Boltzmann select will give all of the 
            individuals a score of e (euler's constant) to the power of the 
            individuals fitness. This score, divided by the sum of all scores, 
            is the probability an individual will be chosen. 
        Input: none 
        Return: none
        """
        selected_individuals = []
        scored_individuals = []
        sum_score = 0
        for i in range (0, len(self.individuals)):
            #NOTE: use percentage correct as measure of fitness here to get around overflow
            perc_fit = self.individuals[i].fitness / MAXSAT_PROBLEM["num_clauses"] * 100
            individ_score = (pow(math.e, perc_fit))
            sum_score += individ_score
            scored_individuals.append((individ_score, i))
        #sort by e^fitness
        scored_individuals = sorted(scored_individuals, key=lambda tup: tup[0])
        for i in range(0, len(self.individuals)):
            bucket_score = 0
            rand_num = random.uniform(0.0, sum_score)
            i = 0
            while bucket_score <= rand_num:
                individ = scored_individuals[i]
                bucket_score += individ[0]
                if rand_num < bucket_score:
                    selected_individuals.append(self.individuals[individ[1]])
                i += 1
        del self.individuals[:]
        self.individuals = selected_individuals

    def single_crossover(self, individual1, individual2): 
        """
        Purpose: From two individuals, use a single crossover point to produce two children.
        Parameters: Two individual classes, each with a MAXSAT solution as a list of Boolean
            values as the solution instance variable.
        Return: A tuple of the two two children produced by the single point crossover.  
        """
        crossover_point = random.randint(1, len(individual1.solution) - 1)
        #first child:
        first_part1 = individual1.solution[:crossover_point].copy()
        first_part2 = individual2.solution[crossover_point:].copy()
        first_child_solution =  first_part1 + first_part2
        #second child:
        second_part1 = individual2.solution[:crossover_point].copy()
        second_part2 = individual1.solution[crossover_point:].copy()
        second_child_solution = second_part1 + second_part2
        first_child = Individual(first_child_solution)
        second_child = Individual(second_child_solution)

        return (first_child, second_child)

    def uniform_crossover(self, individual1, individual2):
        """
        Purpose: Perform uniform crossover on two indiduals; each pair produces two offspring
            by flipping a coin to determine which parent passes down the literal assignment
            to the child
        Parameters: The two individuals that are making children
        Return: A tuple of the first and second children of individual1/2
            Note:
                - everyone has two children to maintain population size
                - the first child is the best of the bunch
        """
        breeding_pair = (individual1, individual2)
        first_child = []
        worse_child = []  # @dgans, @djanderson

        for index in range(0, len(breeding_pair[0].solution)):
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
        individual_first = Individual(first_child)
        individual_second = Individual(worse_child)
        return (individual_first, individual_second)


class Individual:
    """
    Purpose: individual class that includes a get_fitness method to calculate
        the fitness of an individual.
    """
    def __init__(self, bools_array):
        self.solution = bools_array
        self.fitness = -1

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
    """
    Purpose: keep track of the best solution so far and to provide a method to
        compare the best so far to an individual.
    """
    def __init__(self, individual, iteration):
        self.individual = individual
        self.iteration_found = iteration

    def compare_to_best(self, individual, iteration):
        """
        Purpose: Update the best solution so far if the given individual
            has a better solution.
        Input: individual to check against best so far, iteration this individual
            is from.
        Return:  Boolean indicating whether the given individual was better than the
            best solution so far.
        """
        if individual.fitness > self.individual.fitness:
            self.individual.solution = individual.solution.copy()
            self.individual.fitness = individual.fitness
            self.iteration_found = iteration
            print("Found new best with score {} in generation {}".format(
                self.individual.fitness, self.iteration_found))
            return True

        return False


def pretty_solution(solution):
    """
    Purpose: Modify the solution to that it is represented as a mostly legible
        string.
    Input: Solution as a list of boolean values.
    Return: Solution represented as a string
    """
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


def print_solution(best_so_far, parameters):
    """
    Purpose: print output to the specifications of the project writeup.
    Input: object representing the best solution, object representing the 
        problem parameters
    Return: none
    """
    print("File: {}".format(parameters.file_name))
    num_literals = MAXSAT_PROBLEM["num_literals"]
    num_clauses = MAXSAT_PROBLEM["num_clauses"]
    print("Literals count: {}\nClauses count: {}".format(num_literals, num_clauses))
    fitness_div_clauses = best_so_far.individual.fitness / MAXSAT_PROBLEM["num_clauses"]
    percentage_correct = round(fitness_div_clauses * 100, 1)
    print("Best individual scored {} ({}%)".format(best_so_far.individual.fitness,
        percentage_correct))
    print("Difference: {}".format(MAXSAT_PROBLEM["num_clauses"] -
        best_so_far.individual.fitness))
    print("Solution:\n{}".format(pretty_solution(best_so_far.individual.solution)))
    print("Found in iteration {}".format(best_so_far.iteration_found))


def standard_GA(parameters):
    """
    Purpose: Outmost function for the GA.  For iterations equal to the specified
        number of generations, score, select, recombine, and mutate the population
        to work towards the solution to a MAXSAT problem
    Parameters: parameters object including each of the command line arguments
        (these are detailed in the module docstring)
    Return: best solution found (for testing purposes)
    """
    population = Population(0, parameters.pop_size)
    population.generate_initial_pop(MAXSAT_PROBLEM["num_literals"])
    population.individuals[0].get_fitness(MAXSAT_PROBLEM)
    
    #arbitrarily initialize best_so_far
    best_so_far = BestSoFar(population.individuals[0], 0)
    if best_so_far.individual.fitness == MAXSAT_PROBLEM["num_clauses"]:
        print("Full Solution!")
        print_solution(best_so_far, MAXSAT_PROBLEM, parameters)
        return

    iteration = 1
    #print(parameters.num_generations)
    while iteration <= parameters.num_generations:
        print("Generation: {}".format(iteration))
        population.next_generation()
        population.score_individuals(best_so_far)
        # check if we have found a complete solution (not likely):
        if best_so_far.individual.fitness == MAXSAT_PROBLEM["num_clauses"]:
            print("Full Solution!")
            print_solution(best_so_far, MAXSAT_PROBLEM, parameters)
            return
        population.select(parameters.selection_type)        
        population.recombination(parameters.xover_prob, parameters.xover_method)      
        population.mutate(parameters.mutation_prob)
        iteration += 1

    population.score_individuals(best_so_far)
    print_solution(best_so_far, parameters)
    return best_so_far


'''
Function used for testing. Feel free to ignore
def for_testing(file_name, pop_size, selection_type, xover_method, xover_prob, mutation_prob, num_generations, algorithm):
    """
    Used to test in conjuntion with the test module.
    """
    global MAXSAT_PROBLEM
    MAXSAT_PROBLEM = parse.return_problem("test_problems/" + file_name)
    parameters = Parameters(file_name, pop_size, selection_type, xover_method, 
        xover_prob, mutation_prob, num_generations, algorithm)
    start = time.time()
    solution = standard_GA(parameters)
    finished = time.time()
    run_time = finished - start
    #time taken, how good the solution is, generation best solution 
    return (solution, run_time)
'''


def main():
    # acquire command line arguments
    global MAXSAT_PROBLEM
    # Decide how to interpret parameters (based on algo we're using):
    if sys.argv[8] == 'ga':
        parameters = Parameters(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],
            sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8])
        # Round the population size to that things work nicely
        if parameters.pop_size % 2 != 0:
            parameters.pop_size += 1
    elif sys.argv[8] == 'pbil':
        #print("Ind to incl = {}".format(sys.argv[3]))
        parameters = PBIL.PBILParameters(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
            sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
    
    # Acquire MAXSAT problem
    MAXSAT_PROBLEM = parse.return_problem(FILE + parameters.file_name)
    if parameters.algorithm == 'ga':
        standard_GA(parameters)
    else:
        PBIL.pbil(MAXSAT_PROBLEM, parameters)


# Run main like a normal person:
main()
