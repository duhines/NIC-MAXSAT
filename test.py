import os 
import genetic_alg
import xlsxwriter
import pbil as PBIL

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
	"""
	for file in os.listdir("test_problems"):
		if file[0] == ".":
			continue
		print(file)
		for s_option in select_options:
			print(s_option)
			for c_option in crossover_options:
				print(c_option)
				for pop_size in range(1, 6):
					print(pop_size)
					for xover_prob in range(2, 6):
						print(xover_prob)
						for mutation_prob in range (0, 5):	
							parameters = {
								"file_name": file,
								"pop_size": pop_size * 20,
								"selection_type": s_option,
								"xover_method": c_option,
								"xover_prob": (xover_prob/5)-.1,
								"mutation_prob": (mutation_prob/20) + .001,
								"num_generations": 10, 
								"algorithm": "ga"
							}
							solution = genetic_alg.for_testing(file, pop_size * 20, s_option, c_option, (xover_prob / 5) - .1, mutation_prob/20 + .001, 75, 'ga')
							data = {
								"solution": solution,
								"parameters": parameters
							}
							fitness = data["solution"][0].individual.fitness
							print("-{}-".format(fitness))
							datas.append(data)
	"""
	k = 0
	for file in os.listdir("test_problems"):
		if file[0] == ".":
			continue
		else:
			#for select_type in select_options:
				#for c_type in crossover_options:
			for pop in range(0, 1):
				print(k/(15*10))
				k += 1
				solution = genetic_alg.for_testing(file, 100, "bs", "uc", .9, .01, 800, 'ga')
				parameters = {
					"file_name": file,
					"pop_size": 100,
					"selection_type": "ts",
					"xover_method": "uc",
					"xover_prob": .9,
					"mutation_prob": .01,
					"num_generations": 200, 
					"algorithm": "ga"
				}
				data = {
					"solution": solution,
					"parameters": parameters
				}
				datas.append(data)

	workbook = xlsxwriter.Workbook('accuracy_solution2.xlsx')
	worksheet = workbook.add_worksheet()
	file = []
	select = []
	crossover = []
	pop = []
	cross_prob = []
	mut_prob = []
	num_gen = []
	alg = []
	score = []
	iter_found = []
	time = []
	for item in datas:
		param = item["parameters"]
		file.append(param["file_name"])
		select.append(param["selection_type"])
		crossover.append(param["xover_method"])
		pop.append(param["pop_size"])
		cross_prob.append(param["xover_prob"])
		mut_prob.append(param["mutation_prob"])
		num_gen.append(param["num_generations"])
		alg.append(param["algorithm"])
		iter_found.append(item["solution"][0].iteration_found)
		time.append(item["solution"][1])
		score.append(item["solution"][0].individual.fitness)
	array = [file, select, crossover, pop, cross_prob, mut_prob, num_gen, alg, score, iter_found, time]
	row = 0

	for col, data in enumerate(array):
	    worksheet.write_column(row, col, data)

	workbook.close()

def test_like_a_robot():
	datas = []
	for file in os.listdir("test_problems"):
		if file[0] == ".":
			continue
		else:
			for  num_geners in range(50, 5050, 100):
				solution = PBIL.test_pbil(file, 130, 1, 0.17, 0.05, 0.01, num_geners, 'pbil')
				parameters = {
					"file_name": file,
					"pop_size": 130,
					"num_incl": 1,
					"alpha": 0.17,
					"shift": 0.05,
					"mutation_prob": 0.01,
					"num_generations": num_geners,
					"algorithm": "pbil"
				}
				data = {
					"solution": solution,
					"parameters": parameters
				}
				datas.append(data)

	workbook = xlsxwriter.Workbook('num_gens_pbil.xlsx')
	worksheet = workbook.add_worksheet()
	file = []
	pop = []
	num = []
	learning_rate = []
	mu_shift = []
	mut_prob = []
	num_gen = []
	alg = []
	score = []
	iter_found = []
	time = []
	for item in datas:
		param = item["parameters"]
		file.append(param["file_name"])
		pop.append(param["pop_size"])
		num.append(param["num_incl"])
		learning_rate.append(param["alpha"])
		mu_shift.append(param["shift"])
		mut_prob.append(param["mutation_prob"])
		num_gen.append(param["num_generations"])
		alg.append(param["algorithm"])
		iter_found.append(item["solution"][0].iteration_found)
		time.append(item["solution"][1])
		score.append(item["solution"][0].fitness)
	array = [file, pop, num, learning_rate, mu_shift, mut_prob, num_gen, alg, score, iter_found, time]
	row = 0

	for col, data in enumerate(array):
	    worksheet.write_column(row, col, data)

	workbook.close()


def main():	
	#run_as_if_command_line()
	# test_like_a_normal_person()
	test_like_a_robot()

main()
