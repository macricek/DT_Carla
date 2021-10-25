import numpy
import numpy as np
import pygad

function_inputs = [4, -2, 3.5, 5, -11, -4.7]  # Function inputs.
desired_output = 44  # Function output.
num_generations = 1000
num_parents_mating = 4

sol_per_pop = 50
num_genes = len(function_inputs)

init_range_low = -2
init_range_high = 5


def fit(solution, solution_idx):
    output = numpy.sum(solution * function_inputs)
    fitness = 1.0 / numpy.abs(output - desired_output)
    return fitness


def myGenetic():
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           fitness_func=fit,
                           mutation_num_genes=1,
                           #mutation_probability=20,
                           keep_parents=0)
    ga_instance.initialize_population(init_range_low, init_range_high, False, False, float)
    ## calc fitness of solutions
    ga_instance.solutions_fitness = ga_instance.cal_pop_fitness()
    for i in range(0, num_generations):
        ga_instance.last_generation_fitness = ga_instance.solutions_fitness
        ## selections
        population = ga_instance.population
        [workPop1,_] = ga_instance.tournament_selection(ga_instance.solutions_fitness, 20)

        [workPop2,_] = ga_instance.random_selection(ga_instance.solutions_fitness, 20)
        [justBest,_] = ga_instance.rank_selection(ga_instance.solutions_fitness, 9)
        [bestOne,bestIndices] = ga_instance.rank_selection(ga_instance.solutions_fitness, 1)

        ## operations
        workPop1 = ga_instance.two_points_crossover(workPop1, (workPop1.shape[0], workPop1.shape[1]))
        ga_instance.mutation_probability = 0.1
        workPop2 = ga_instance.random_mutation(workPop2)
        ga_instance.mutation_probability = 0.05
        bestOne = ga_instance.random_mutation(bestOne)

        ## connect pop tgthr
        ga_instance.population = np.concatenate((workPop1, workPop2, justBest, bestOne))

        ## calc fitness of solutions
        ga_instance.solutions_fitness = ga_instance.cal_pop_fitness()
        [ga_instance.best_solution_generation, bestFit, _] = ga_instance.best_solution(ga_instance.solutions_fitness)
        ga_instance.best_solutions_fitness.append(bestFit)
        ga_instance.generations_completed += 1


    print("-------------MY--------------")
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    prediction = numpy.sum(numpy.array(function_inputs) * solution)
    print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
    print(ga_instance.best_solutions_fitness)
    return ga_instance, solution_fitness


# num parents -> members in one generation
#
def genetic():
    parent_selection_type = "sss"
    keep_parents = 1
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 10

    ga_instance = pygad.GA(num_generations=1000,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fit,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           crossover_probability=0.1,
                           mutation_type=mutation_type,
                           mutation_num_genes=1)

    ga_instance.run()
    print("-------------AUTO--------------")
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    prediction = numpy.sum(numpy.array(function_inputs) * solution)
    print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
    print(ga_instance.best_solutions_fitness)
    #ga_instance.plot_fitness()
    return ga_instance, solution_fitness

