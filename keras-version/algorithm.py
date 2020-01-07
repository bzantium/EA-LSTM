import random
import math
from itertools import combinations


def initialize_population(pop_size, time_steps, code_length):
    pop = []
    for _ in range(pop_size):
        indiv = []
        for _ in range(time_steps * code_length):
            indiv.append(1 if random.random() > 0.5 else 0)
        pop.append(indiv)
    return pop


def decode(code):
    code_length = len(code)
    code = [str(i) for i in code]
    # transform binary system to decimal system
    element = round(int(''.join(code), 2) / (math.pow(2, code_length) - 1), 2)
    return element


def get_weight(indiv, time_steps, code_length):
    weight = []
    for i in range(time_steps):
        weight.append(decode(indiv[i * code_length: (i + 1) * code_length]))
    return weight


def pop_to_weights(pop, time_steps, code_length):
    weights = []
    for indiv in pop:
        weights.append(get_weight(indiv, time_steps, code_length))
    return weights


def initialize_weights(pop_size, time_steps, code_length):
    pop = initialize_population(pop_size, time_steps, code_length)
    weights = []
    for indiv in pop:
        weights.append(get_weight(indiv, time_steps, code_length))
    return pop, weights


def get_segment_ids(indiv_length):
    index = []
    while True:
        for i in range(indiv_length):
            if random.random() > 0.5:
                index.append(i)
        if len(index) > 0:
            break
    return index


def crossover(indiv1, indiv2):
    indiv_length = len(indiv1)
    a_index = get_segment_ids(indiv_length)
    b_index = []
    for i in range(indiv_length):
        if i not in a_index:
            b_index.append(i)

    new = list()
    for i in range(indiv_length):
        new.append(0)

    for i in a_index:
        new[i] = indiv1[i]
    for i in b_index:
        new[i] = indiv2[i]

    if random.random() > 0.8:
        new = mutation(new)
    return new


def mutation(indiv):
    indiv_length = len(indiv)
    index = get_segment_ids(indiv_length)
    for i in index:
        if indiv[i] == 0:
            indiv[i] = 1
        else:
            indiv[i] = 0
    return indiv


def group_population(pop, n_group):
    assert len(pop) % n_group == 0, "pop_size must be a multiple of n_group."
    per_group = len(pop) // n_group
    group_index = list(range(0, len(pop)))
    random.shuffle(group_index)
    group_pop = []
    for i in range(n_group):
        temp_index = group_index[i * per_group: (i + 1) * per_group]
        temp_pop = []
        for j in temp_index:
            temp_pop.append(pop[j])
        group_pop.append(temp_pop)
    return group_pop


def individual_to_key(indiv):
    temp = [str(i) for i in indiv]
    key = ''.join(temp)
    return key


def select(pop, n_select, key_to_rmse):
    group_pop = group_population(pop, n_select)
    fitness_selected = []
    pop_selected = []
    for sub_group in group_pop:
        fitness = []
        for indiv in sub_group:
            key = individual_to_key(indiv)
            fitness.append(key_to_rmse[key])
        min_fitness = min(fitness)
        pop_selected.append(sub_group[fitness.index(min_fitness)])
        fitness_selected.append(min_fitness)

    return pop_selected, fitness_selected


def reconstruct_population(pop_selected, pop_size):
    new_pop = list()
    new_pop.extend(pop_selected)
    pop_map = {}
    for i in range(len(new_pop)):
        pop_map[individual_to_key(new_pop[i])] = i

    index = [c for c in combinations(range(len(pop_selected)), 2)]
    while len(new_pop) < pop_size:
        for combi in index:
            new_indiv = crossover(pop_selected[combi[0]], pop_selected[combi[1]])
            if not individual_to_key(new_indiv) in pop_map.keys():
                new_pop.append(new_indiv)
                pop_map[individual_to_key(new_indiv)] = len(new_pop)
            if len(new_pop) == pop_size:
                break
    return new_pop
