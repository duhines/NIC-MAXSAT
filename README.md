# NIC-MAXSAT
Project 1 for Nature Inspired Computation (fall 2018).  This project compares a genetic algorithm with a population-based incremental learning algorithm in solving MAXSAT problems.   

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
