import numpy as np
import pandas as pd
import math 
from copy import deepcopy
from Fitness import quality_prediction, fitnessTestData
from sklearn.ensemble import RandomForestClassifier

class GeneticProgramming:

    TERMINALS = []
    
    def __init__(
            self,
            fitness_function,
            offspring_size,
            num_solutions_tournament,
            mutation_probability,
            fitnessFunctionData,
            testData,
            population_size
    ):
        self.fitness_function = fitness_function
        self.offspring_size = offspring_size
        self.mutation_probability = mutation_probability
        self.num_solutions_tournament = num_solutions_tournament
        self.fitnessFunctionData = fitnessFunctionData
        self.population_size = population_size
        self.TERMINALS = list(self.fitnessFunctionData.columns.array)
        self.TERMINALS.remove("sequence")
        self.TERMINALS.remove("label")
        self.testData = testData

    class GPSolution:

        def __int__(self):
            #lista stabala (GPTree)
            self.features = []

            #dataset jedinke
            self.dataset = 0
            self.fitness = -1

            # Dataset made with test data
            self.testDataset = None
            # Fitness score achived on the test data
            self.testFitness = -1

            self.classifierRndForest = None

            #Indexi featurea na koje ce djelovati evolucijski proces
            self.evolvingFeatures = None

            #Indexi feature na koje nece djelovati evolucijski proces
            self.hiddenFeatures = None


    class GPTree:
        MIN_DEPTH = 2
        MAX_DEPTH = 5
        
        def addition(x, y):
            return x + y

        def difference(x, y):
            return x - y

        def multiplication(x, y):
            return x * y
        
        def division(x, y):
            if y == 0:
                return 0
            else:
                return x / y
        # TODO: fixat
        def logarithm(x, y):
            #return math.log(x, y)
            return x

        def lessThan(x, y):
            if x <= y:
                return x
            else:
                return y
            
        def greaterThan(x, y):
            if x >= y:
                return x
            else:
                return y
        
        def equal(x, y):
            if x == y:
                return 1
            else:
                return 0
        
        FUNCTIONS = [addition, difference, multiplication, division,
                     logarithm, lessThan, greaterThan, equal]
        TERMINALS = []


        def __init__(self, 
                     data=None,
                     left_node=None, 
                     right_node=None, 
                     GP_class=None, 
        ):
            self.data = data
            self.left_node = left_node
            self.right_node = right_node
            self.GP_class = GP_class
            self.TERMINALS = list(GP_class.fitnessFunctionData.columns.array)
            self.TERMINALS.remove("sequence")
            self.TERMINALS.remove("label")
        
        def print_tree(self, prefix=""):
            print("%s%s" % (prefix, self.node_label()))

            if self.left_node:
                self.left_node.print_tree(prefix+"   ")

            if self.right_node:
                self.right_node.print_tree(prefix+"   ")

        def node_label(self):
            if self.data in GeneticProgramming.GPTree.FUNCTIONS:
                return self.data.__name__
            else:
                return str(self.data)

        # Bio je x u argumentu funkccije, funkcija bi trebala krenut od roota
        # i spustat se niz stablo
        #redak je jedan red iz inicijalnog dataseta, tip je pandas series
        def compute_tree(self, redak):
            if self.data in GeneticProgramming.GPTree.FUNCTIONS:
                return self.data(
                    self.left_node.compute_tree(redak),
                    self.right_node.compute_tree(redak)
                )
            else:
                return redak[self.data]

        def random_tree(self, grow, max_depth, depth=0):
            if depth < GeneticProgramming.GPTree.MIN_DEPTH \
                    or (depth < max_depth and not grow):

                self.data = GeneticProgramming.GPTree.FUNCTIONS[
                                np.random.randint(
                                    len(GeneticProgramming.GPTree.FUNCTIONS)
                                )
                            ]

            elif depth >= max_depth:
                self.data = self.GP_class.TERMINALS[
                                np.random.randint(
                                    len(self.GP_class.TERMINALS)
                                )
                            ]

            else:
                if np.random.rand() > 0.5:
                    self.data = self.GP_class.TERMINALS[
                                    np.random.randint(
                                        len(self.GP_class.TERMINALS)
                                    )
                                ]
                else:
                    self.data = GeneticProgramming.GPTree.FUNCTIONS[
                                    np.random.randint(
                                        len(GeneticProgramming.GPTree.FUNCTIONS)
                                    )
                                ]

            if self.data in GeneticProgramming.GPTree.FUNCTIONS:
                self.left_node = GeneticProgramming.GPTree(GP_class=self.GP_class)
                self.left_node.random_tree(grow, max_depth, depth=depth+1)

                self.right_node = GeneticProgramming.GPTree(GP_class=self.GP_class)
                self.right_node.random_tree(grow, max_depth, depth=depth+1)
                
        #line 134- grow ?
        def mutation(self):
               if np.random.rand() < self.GP_class.mutation_probability:
                      self.random_tree(grow=False, max_depth=3)
              
               elif self.left_node:
                      self.left_node.mutation()
              
               elif self.right_node:
                      self.right_node.mutation()

               # Other je drugi roditelj

        def crossover(self, other):
            second = other.scan_tree(np.random.randint(other.size()), None)
            self.scan_tree(np.random.randint(self.size()), second)

            # TODO: Prepraviti sve
            # Broj cvorova u stablu

        def size(self):
            if self.data in GeneticProgramming.GPTree.TERMINALS:
                return 1

            left_size = self.left_node.size() if self.left_node else 0
            right_size = self.right_node.size() if self.right_node else 0

            return left_size + right_size + 1

        def build_subtree(self):
            tree = GeneticProgramming.GPTree(GP_class=self.GP_class)
            tree.data = self.data

            if self.left_node:
                tree.left_node = self.left_node.build_subtree()

            if self.right_node:
                tree.right_node = self.right_node.build_subtree()

            return tree

        def scan_tree(self, count, second):
            count -= 1

            if count <= 1:
                if not second:
                    return self.build_subtree()
                else:
                    self.data = second.data
                    self.left_node = second.left_node
                    self.right_node = second.right_node

            else:
                ret = None
                if self.left_node and count > 1:
                    ret = self.left_node.scan_tree(count, second)

                if self.right_node and count > 1:
                    ret = self.right_node.scan_tree(count, second)

                return ret

        def mutation(self):
            if np.random.rand() < self.GP_class.mutation_probability:
                self.random_tree(grow=True, max_depth=2)

            elif self.left_node:
                self.left_node.mutation()

            elif self.right_node:
                self.right_node.mutation()


    def generate_random_population(self):
            # Randomly generate initial population. The trees in the population will
            # have different depths.

            #array populacije
            pop = []

            #for md in range(3, GeneticProgramming.GPTree.MAX_DEPTH + 1):

            ## Iterira population.size puta
            for _ in range(self.population_size):
                jedinka = GeneticProgramming.GPSolution()

                # Radnom generiranje listi, hiddenFeatures i evolvingFeatures
                jedinka.hiddenFeatures = list()
                brojac = 0
                while brojac < 3:
                    randomIndex = np.random.randint(0, 10)
                    if randomIndex not in jedinka.hiddenFeatures:
                        jedinka.hiddenFeatures.append(randomIndex)
                        brojac += 1

                jedinka.evolvingFeatures = list()
                brojac = 0
                while brojac < 7:
                    randomIndex = np.random.randint(0, 10)
                    if randomIndex not in jedinka.evolvingFeatures and randomIndex not in jedinka.hiddenFeatures:
                        jedinka.evolvingFeatures.append(randomIndex)
                        brojac += 1

                ## Iterira 10 puta, jer jedna jedinka ima 10 stabala
                features = []

                for i in range(10):
                    t = GeneticProgramming.GPTree(GP_class=self)
                    t.random_tree(grow=True, max_depth=t.MAX_DEPTH)
                    features.append(t)

                jedinka.features = features.copy()
                pop.append(jedinka)

            return pop

    def calculateTestFitness(self, population):

        # za svaki solution izracunaj fitnes
        i = 0
        for solution in population:
            i += 1
            # Columns for new dataset
            columns = ["sequence", "label", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]
            # newDataset = pd.DataFrame(columns = columns)

            # Retci koji ce se koristiti za novi dataset dataframe
            newDatasetRows = []

            # Iteriaj korz svaki redak inicjalnog dataseta
            for index, redak in self.testData.iterrows():
                # novi redak, dodaj mu sekvencu
                newRow = []
                newRow.append(redak["sequence"])
                newRow.append(redak["label"])

                # dropaj label i squence iz redka
                red = redak.copy()
                red = red.drop(["label", "sequence"])

                # Za svaki feature (GPTree)
                for feature in solution.features:
                    newRow.append(feature.compute_tree(red))

                newDatasetRows.append(newRow)

            newDataset = pd.DataFrame(columns=columns, data=newDatasetRows)
            solution.testDataset = newDataset
            solution.testFitness = fitnessTestData(solution.testDataset, solution.classifierRndForest)
            #print("Jedinka #" + str(i) + " -- Ftiness Test data: " + str(solution.testFitness))

    def calculate_fitness_scores(self, population):
        # Calculates and returns fitness scores of each individual in the
        # population.

        # za svaki solution izracunaj fitnes
        for solution in population:
            #TODO: Pogledati koliko feature-a treba generirati? 10
            #Columns for new dataset
            columns = ["sequence", "label", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]
            #newDataset = pd.DataFrame(columns = columns)

            #Retci koji ce se koristiti za novi dataset dataframe
            newDatasetRows = []

            # Iteriaj korz svaki redak inicjalnog dataseta
            for index, redak in self.fitnessFunctionData.iterrows():
                #novi redak, dodaj mu sekvencu
                newRow = []
                newRow.append(redak["sequence"])
                newRow.append(redak["label"])

                # dropaj label i squence iz redka
                red = redak.copy()
                red = red.drop(["label", "sequence"])

                #Za svaki feature (GPTree)
                for feature in solution.features:
                    newRow.append(feature.compute_tree(red))

                newDatasetRows.append(newRow)

            newDataset = pd.DataFrame(columns = columns, data = newDatasetRows)
            solution.dataset = newDataset
            solution.fitness, featureImportances, solution.classifierRndForest = quality_prediction(solution.dataset)


            featureImportances = enumerate(featureImportances)
            featureImportances = sorted(featureImportances, key=lambda feature:feature[1])

            hidden = featureImportances[-3:]
            evol = featureImportances[0:7]

            solution.hiddenFeatures.clear()
            for index, importance in hidden:
                solution.hiddenFeatures.append(index)

            solution.evolvingFeatures.clear()
            for index, importance in evol:
                solution.evolvingFeatures.append(index)



            # for indexH, hiddenFeature  in enumerate(solution.hiddenFeatures):
            #     for indexE, evolvingFeature in enumerate(solution.evolvingFeatures):
            #         if featureImportances[evolvingFeature] > featureImportances[hiddenFeature]:
            #             solution.hiddenFeatures[indexH], solution.evolvingFeatures[indexE] = solution.evolvingFeatures[indexE], solution.hiddenFeatures[indexH]
            #
            #
            # print(featureImportances)
            # print(solution.hiddenFeatures)
            # print(solution.evolvingFeatures)

    def select_parent(self, population, fitness_scores):
        # Performs turnament selection with self.num_solutions_tournament
        # individuals.

        tournament = np.random.randint(
                            0,
                            len(population),
                            self.num_solutions_tournament)
        tournament_fitness_scores = [
            fitness_scores[solution_index] for solution_index in tournament
        ]

        return deepcopy(
            population[
                tournament[
                    tournament_fitness_scores.index(max(tournament_fitness_scores))
                ]
            ]
        )

    def generate_offspring(self, population, fitness_scores):
        # Generates and returns offspring.

        offspring = []

        for _ in range(self.offspring_size):
            #select_parent vraca kopiju
            parent1 = self.select_parent(population, fitness_scores)
            parent2 = self.select_parent(population, fitness_scores)

            # Radi crossover sa svim znacajkama unutar evolving listi
            for parent1Evol, parent2Evolv in zip(parent1.evolvingFeatures, parent2.evolvingFeatures):
                parent1.features[parent1Evol].crossover(parent2.features[parent2Evolv])
                parent1.features[parent1Evol].mutation()

            # for i in range(len(parent1.features)):
            #     parent1.features[i].crossover(parent2.features[i])
            #     parent1.features[i].mutation()

            #parent1.crossover(parent2)
            #parent1.mutation()


            offspring.append(parent1)

        return offspring

    def next_generation(self, population, fitness_scores):
        # Generates and returns offspring.

        merged_list = list(zip(population, fitness_scores))
        merged_list.sort(key=lambda member: member[1])

        sorted_population, _ = zip(*merged_list)

        return list(sorted_population[-self.population_size:])