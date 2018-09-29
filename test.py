
import os 
import genetic_alg

def run_as_if_command_line():
	os.system("python3 genetic_alg.py maxcut-140-630-0.8-34.cnf 500 ts 1c .5 .5 100 ga")

def test_like_a_normal_person():
	"""	
		for_testing(file_name, pop_size, selection_type, xover_method, xover_prob, mutation_prob, num_generations, algorithm)
		return (solution, run_time)
	"""
	"""
	print("ASDFAFSD")
	print(genetic_alg.for_testing("maxcut-140-630-0.7-1.cnf", 500, "ts", "uc", .6, .01, 10, 'ga')[0].individual.fitness)
	print("right before me")
	return
	"""
	select_options = ["ts", "rs", "bs"]
	crossover_options = ["1c", "uc"]
	datas = []
	for file in os.listdir("test_problems"):
		print(file)
		for s_option in select_options:
			print(s_option)
			for c_option in crossover_options:
				print(c_option)
				for pop_size in range(1, 5):
					print(pop_size)
					for xover_prob in range(1, 5):
						print(xover_prob)
						for mutation_prob in range (5, 10):	
							print("       {}".format(mutation_prob))
							parameters = {
								"file_name": file,
								"pop_size": pop_size * 20,
								"selection_type": s_option,
								"xover_method": c_option,
								"xover_prob": xover_prob/10,
								"mutation_prob": mutation_prob/1000,
								"num_generations": 10, 
								"algorithm": "ga"
							}
							print(parameters)
							solution = genetic_alg.for_testing(file, pop_size * 20, s_option, c_option, xover_prob / 5, mutation_prob / 500, 10, 'ga')
							data = {
								"solution": solution,
								"parameters": parameters
							}
							fitness = data["solution"][0].individual.fitness
							print("-{}-".format(fitness))
							datas.append(data)

	print(datas)
def main():	
	#run_as_if_command_line()
	test_like_a_normal_person()


main()