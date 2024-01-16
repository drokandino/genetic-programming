from GeneticProgramming import GeneticProgramming
import Fitness
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle


def createFitnessList(population):
    fitnessScores = list()
    testFitnessSocres = list()

    for jedinka in population:
        fitnessScores.append(jedinka.fitness)
        testFitnessSocres.append(jedinka.testFitness)

    return fitnessScores, testFitnessSocres


#data = pd.read_csv("avpdb.csv")
data = pd.read_csv("amp.csv")
data = shuffle(data, random_state=5)

# Data split
dataTrain, dataTest = train_test_split(data, stratify=data["label"], test_size=0.1, random_state=42)

# gp = GeneticProgramming(
#     fitness_function = Fitness.quality_prediction,
#     offspring_size = 100,
#     num_solutions_tournament = 100,
#     mutation_probability = 0.15,
#     fitnessFunctionData = data,
#     population_size = 500
#     )

gp = GeneticProgramming(
    fitness_function = Fitness.quality_prediction,
    offspring_size = 20,
    num_solutions_tournament = 20,
    mutation_probability = 0.15,
    fitnessFunctionData = dataTrain,
    testData=dataTest,
    population_size = 100
    )



# Random pocetna populacija
population = gp.generate_random_population()
print("Generated random population")

# Fitness populacije
gp.calculate_fitness_scores(population)
gp.calculateTestFitness(population)

# Print mcc for initial dataset with original features
print("Original dataset mcc: " + str(Fitness.quality_prediction(data)[0]))

# Loop kroz 50 generacija
for i in range(50):

    fitnessScores, testFitnessScores = createFitnessList(population)

    # Rekombinacija
    offspring = gp.generate_offspring(population, fitnessScores)
    gp.calculate_fitness_scores(offspring)

    population += offspring

    gp.calculateTestFitness(population)


    fitnessScores, testFitnessScores = createFitnessList(population)
    bestSolutionIndex = fitnessScores.index(max(fitnessScores))
    bestSolution = population[bestSolutionIndex]


    population = gp.next_generation(population, fitnessScores)

    print("GENERATION ------------------------------------------ " + str(i+1))
    print("Max fitness on validation data: " + str(max(fitnessScores)))
    print("Min fitness on validation data: " + str(min(fitnessScores)))
    print("Max fitness on test data: " + str(max(testFitnessScores)))
    print("Min fitness on test data: " + str(min(testFitnessScores)))
    print("")
    #print(bestSolution.fitness)



# for i in range(10):
#     print("Generacija " + str(i))
#     fitnessScores.clear()
#     for jedinka in offspring:
#         fitnessScores.append(jedinka.fitness)
#
#     offspring = gp.generate_offspring(offspring, fitnessScores)
#     gp.calculate_fitness_scores(offspring)
#print(gp.select_parent(population, fitnessScores))
#print(population[0].features[1].print_tree())


