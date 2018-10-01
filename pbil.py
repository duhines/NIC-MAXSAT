import random
import time
import parse_input as parse

FILE = "problems/"
MAXSAT_PROBLEM = []


class PBILParameters:
    """
    Class used to store PBIL cmd line args conveniently.
    """
    def __init__(self, file_name, pop_size,
                 ind_to_include, alpha,
                 mutation_shift, mutation_prob,
                 num_generations, algorithm):
        self.file_name = file_name
        self.pop_size = int(pop_size)
        self.ind_to_incl = int(ind_to_include)
        self.alpha = float(alpha)
        self.mu_shift = float(mutation_shift)
        self.mutation_prob = float(mutation_prob)
        self.num_generations = int(num_generations)
        self.algorithm = algorithm
        

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
                individual.append(random.random() < self.vector[j])
            population.append(individual.copy())
            individual = []
        return population

    # A method to update the probability vector based on the N best individuals
    def update_vector(self, scored_pop, ind_to_incl, alpha):
        for individual in scored_pop[0:ind_to_incl]:
            for i in range(self.num_lits):
                if individual[0][i]:
                    self.vector[i] = self.vector[i]*(1 - alpha) + alpha
                else:
                    self.vector[i] = self.vector[i]*(1 - alpha)

    # Mutate pop vector. Mu := P(mutation). Shift := degree of mutation
    def mutate_vector(self, mu, shift):
        for i in range(len(self.vector)):
            if random.random() < mu:
                if random.random() < 0.5:
                    self.vector[i] += shift
                    if self.vector[i] > 1.0:
                        self.vector[i] = 1.0
                elif random.random() >= 0.5 and not self.vector[i] <= 0.0:
                    self.vector[i] -= shift
                    if self.vector[i] < 0.0:
                        self.vector[i] = 0.0

class BestSoFar:
    """
    Purpose: keep track of the best solution so far and to provide a method to
        compare the best so far to an individual.
    """
    def __init__(self, individual, iteration):
        self.individual = individual
        self.iteration_found = iteration
        self.fitness = 0

    def compare_to_best(self, individual, ind_fit, iteration):
        """
        Purpose: Update the best solution so far if the given individual
            has a better solution.
        Input: individual to check against best so far, iteration this individual
            is from, fitness of this individual
        Return:  Boolean indicating whether the given individual was better than the
            best solution so far.
        """
        if ind_fit > self.fitness:
            self.individual = individual.copy()
            self.fitness = ind_fit
            self.iteration_found = iteration
            print("Found new best with score {} in generation {}".format(
                self.fitness, self.iteration_found))
            return True

        return False


def score_pop(population, problem):
    scored_generation = []
    for individual in population:
        score = fitness(individual, problem)
        scored_generation.append((individual, score))
    # From https://stackoverflow.com/questions/3121979/:
    return sorted(scored_generation, key=lambda tup: tup[1], reverse=True)


def fitness(individual, problem):
    """
    Score the fitness of an indivdual based on a MAXSAT problem.
    :param individual: An "individual" represented as an array
    :param problem: MAXSAT problem to compute fitness in ref to, usually
    stored as global MAXSAT_PROBLEM
    :return: An int representation of individuals fitness - higher is better
    """
    fit_score = 0
    for clause in problem["clauses"]:
        check = False
        for literal in clause:
            if literal > 0:
                check = check or individual[literal - 1]
            else:
                check = check or not individual[abs(literal) - 1]
        if check:
            fit_score += 1
    return fit_score


def prettify(individual):
    pretty = ""
    ith_literal = 1
    ten_per_line = 0
    for literal in individual:
        pretty = pretty + "L" + str(ith_literal) + ": " + str(literal) + "  "
        ith_literal = ith_literal + 1
        ten_per_line = ten_per_line + 1
        if ten_per_line > 10:
            ten_per_line = 0
            pretty = pretty + "\n"
    return pretty


def print_PBIL_solution(curr_best, parameters, problem):
    """
    Purpose: Print output in our nice lil standardized way; see writeup
    :param curr_best: List representing the best solution
    :param parameters: Problem parameters we got earlier
    :return: None, this is a printing function.
    """
    print("File: {}".format(parameters.file_name))
    num_literals = problem["num_literals"]
    num_clauses = problem["num_clauses"]
    print("Literals count: {}\nClauses count: {}".format(num_literals, num_clauses))
    fitness_div_clauses = curr_best.fitness / problem["num_clauses"]
    percentage_correct = round(fitness_div_clauses * 100, 1)
    print("Best individual scored {} ({}%)".format(curr_best.fitness,
        percentage_correct))
    print("Difference: {}".format(problem["num_clauses"] -
        curr_best.fitness))
    print("Solution:\n{}".format(prettify(curr_best.individual)))
    print("Found in iteration {}".format(curr_best.iteration_found))


def test_pbil(file_name, pop_size, num_incl, alpha, shift, mutation_prob, num_generations, algorithm):
    """
    Used to test in conjuntion with the test module.
    """
    global MAXSAT_PROBLEM
    MAXSAT_PROBLEM = parse.return_problem("testy/" + file_name)
    parameters = PBILParameters(file_name, pop_size, num_incl, alpha, shift, mutation_prob, num_generations, algorithm)
    start = time.time()
    solution = pbil(MAXSAT_PROBLEM, parameters)
    finished = time.time()
    run_time = finished - start
    #time taken, how good the solution is, generation best solution
    return (solution, run_time)

def pbil(problem, parameters):
    """
    Purpose: This is a function implementing PBIL optimization of MAXSAT
    problems
    :param problem: the MAXSAT problem to optimize, as parsed in parse_input.py
    :param parameters: Problem parameters. Acquired in main of genetic_alg.py
    :return: Returns the best individual found
    """
    pop_vector = PopulationVector(problem["num_literals"])
    curr_best = BestSoFar([], 0)
    # The following is the actual PBIL algorithm:
    iteration = 0
    while iteration < parameters.num_generations:
        #print("Generation: {}".format(iteration))
        nth_pop = pop_vector.generate_population(parameters.pop_size)
        nth_pop = score_pop(nth_pop, problem)

        if iteration == 0:
            curr_best = BestSoFar(nth_pop, iteration)

        # Pull out the best individual and update best_so_far if it's better
        curr_best.compare_to_best(nth_pop[0][0], nth_pop[0][1], iteration)

        # Update pop vector using CSL approach described in paper:
        pop_vector.update_vector(nth_pop,
                                 parameters.ind_to_incl,
                                 parameters.alpha)
        pop_vector.mutate_vector(parameters.mutation_prob,
                                 parameters.mu_shift)
        iteration += 1

    # Final population vector (hopefully) will approximate correct solution
    # So, we round it out and see if it's better than individuals we've already
    # checked.
    final_pop = [round(x) for x in pop_vector.vector]
    curr_best.compare_to_best(final_pop, fitness(final_pop, problem),
                              parameters.num_generations)

    # Return a tuple containing both the rounded "final pop vector"
    # and the actual pop_vector. If the algo worked well, these should
    # be extremely close. If they aren't, then probably have to run longer.
    print_PBIL_solution(curr_best, parameters, problem)
    return curr_best


