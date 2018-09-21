"""
1. Get the parser going.
	i. read cnf file->determine number of literals for the size of the individuals representation 
	ii. put the file in an array, where each item in array is one clause from the file (one line)
		a. clause ex:  ..., 8, 3, -4,... (negative values indicate that the literal is negated) 
	iii. save the boolean values as strings




"""
"""
    STYLE: 
        - 80 chars max per line
        - lets use underscores instead of camel case
        - function docstrings inside functions
        - module docstring at top of each file with 
            - authors, purpose, bugs, etc.
"""
import parse_input as parse
import sys
import random


file = "maxsat-problems/maxsat-crafted/MAXCUT/SPINGLASS/t3pm3-5555.spn.cnf"
def generate_initial_pop(problem, pop_size):
	"""
	Purpose:
	Input: 
	Return:
	"""
	population = []
	individual = []
	for i in range(0, pop_size):
		for j in range(0, problem["num_literals"]):
			individual.append(random.random() > .5)
		population.append(individual.copy())
		del individual[:]

	return population


def fitness(individual, problem):
	"""
	Purpose:
	Input: 
	Return:
	"""
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




def rank_select(scored_generation):
	#thanks: https://stackoverflow.com/questions/3121979/how-to-sort-list-tuple-of-lists-tuples
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

	print(selected_individuals)
	return selected_individuals


def tournament_select():
	return


def boltzmann_select():
	return


def select(scored_generation, selection_method):
	if selection_method == "rs":
		return rank_select(scored_generation)
	elif selection_method == "ts":
		return tournament_select(scored_generation)
	elif selection_method == "bs":
		return boltzmann_select(scored_generation)

def recombination(selected, prob, type):
	next_gen = []
	for i in range (0, len(selected)//2):
		first = selected[random.randint(0, len(selected) - 1)]
		second = selected[random.randint(0, len(selected) - 1)]
		if random.random() < prob:
			if type == "1c":
				offspring = single_crossover(first, second)
			else:
				offspring = uniform_crossover(first, second)
			next_gen.append(offspring[0])
			next_gen.append(offspring[1]) 
		else:
			next_gen.append(first)
			next_gen.append(second)
	return next_gen


def single_crossover(individual1, individual2):
	print("crossover\n")
	print(individual1)
	print(individual2)
	crossover_point = random.randint(1, len(individual1)-2)
	print(crossover_point)
	first_child = individual1[:crossover_point].copy() + individual2[crossover_point:].copy()
	second_child = individual2[:crossover_point].copy() + individual1[crossover_point:].copy()
	print(first_child)
	print(second_child)
	return (first_child, second_child)

def uniform_crossover(individual1, individual2):
	return

def mutate(population):
	return


def standard_GA(problem, parameters):
	initial_generation = generate_initial_pop(problem, parameters["pop_size"])

	for individual in initial_generation:
		print(individual)
		print(fitness(individual, problem))
	iteration = 0
	current_generation = initial_generation.copy()
	while iteration < parameters["num_generations"]:
		iteration += 1
		scored_generation = []
		for individual in current_generation:
			score = fitness(individual, problem)
			scored_generation.append((individual, score))
		selected = select(scored_generation, parameters["selection_type"])
		recombinated_generation = recombination(selected, parameters["crossover_prob"], parameters["crossover_method"])
	return


def main():
	#aquire command line arguments
	parameters = {
		"file_name": sys.argv[1],
		"pop_size": int(sys.argv[2]),
		"selection_type": sys.argv[3],
		"crossover_method": sys.argv[4],
		"crossover_prob": float(sys.argv[5]),
		"mutation_prob": float(sys.argv[6]),
		"num_generations": int(sys.argv[7]),
		"algorithm": sys.argv[2]
	}
	if parameters["pop_size"] % 2 != 0:
		parameters["pop_size"] += 1
	problem = [[1, -4], [-2, -3], [4, 1], [-4, 4], [-3, 1], [-1, 2], [1, 1], [-1, -1]]
	sample_problem = {
    	"num_literals": 4,
    	"num_clauses": 8,
    	"clauses": problem
    }
	#aquire MAXSAT problem
	problem = parse.return_problem(file)
	
	solution = standard_GA(sample_problem, parameters)

	print(problem["num_literals"])
	

main()