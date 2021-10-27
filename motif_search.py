import search         # import the c++ module of incsearch
import numpy as np
import copy
import scipy.sparse as sp
from utils import *

# The largest size of search subgraphs 
max_node = 5 

# Intialize the gene pool of subgraph combinations
def motif_initiate(numGenes, flag_dir, mutate_run):
	gene_list = []
	flag = 0
	for i in range(numGenes):
		gene_list.append([None, np.zeros((2, 2), dtype=np.int32)])
		if flag_dir:
			if flag==0:
				gene_list[-1][1][0, 1] = 1
				flag = 1
			else:
				gene_list[-1][1][1, 0] = 1
				flag = 0				
		else:
			gene_list[-1][1][0, 1] = 1
			gene_list[-1][1][1, 0] = 1
	for i in range(mutate_run):
		gene_list = motif_mutate(gene_list, 0.3, 0.2, 0.8, flag_dir)

	return gene_list

# Intialize the gene pool of subgraph combinations, and return the corresponding adj matrix
def motif_initiate(numGenes, flag_dir, flag_acc, adj_dic, search_base, mutate_run):
	gene_list = []
	flag = 0
	for i in range(numGenes):
		gene_list.append([None, np.zeros((2, 2), dtype=np.int32)])
		if flag_dir:
			if flag==0:
				gene_list[-1][1][0, 1] = 1
				flag = 1
			else:
				gene_list[-1][1][1, 0] = 1
				flag = 0				
		else:
			gene_list[-1][1][0, 1] = 1
			gene_list[-1][1][1, 0] = 1
	for i in range(mutate_run):
		gene_list = motif_mutate(gene_list, 0.3, 0.2, 0.8, flag_dir)
		_, adj_dic = construct_motif_adj_batch([gene_list], adj_dic, search_base, flag_dir, flag_acc)

	return gene_list, adj_dic

# Mutate children subgraphs from given parent subgraphs
def motif_mutate(origGenes, probMutate, probNodes, probEdges, flag_dir):
	mutated_genes = []
	for gene in origGenes:
		if np.random.rand() < probMutate:
			if np.random.rand() < probNodes:
				if len(gene[1]) == max_node:
					mutated_genes.append(gene)
				else:
					gene_plus = np.zeros((len(gene[1])+1, len(gene[1])+1), dtype=np.int32)
					gene_plus[:-1, :-1] = gene[1]
					attach_node = np.random.randint(len(gene[1]))
					if flag_dir:
						if np.random.randint(2):
							gene_plus[-1, attach_node] = 1
						else:
							gene_plus[attach_node, -1] = 1
					else:
						gene_plus[-1, attach_node] = 1
						gene_plus[attach_node, -1] = 1
					mutated_genes.append((gene[1], gene_plus))

			else:
				zeroList = np.where(gene[1]==0)
				edgeChoices = [(zeroList[0][ind], zeroList[1][ind]) for ind in range(len(zeroList[0])) if zeroList[0][ind]!=zeroList[1][ind]]
				if len(edgeChoices) > 0:
					draw = np.random.choice(range(len(edgeChoices)), replace=False)
					gene_plus = np.zeros(gene[1].shape, dtype=np.int32)
					gene_plus[:,:] = gene[1] 
					if flag_dir:
						gene_plus[edgeChoices[draw][0], edgeChoices[draw][1]] = 1
					else:
						gene_plus[edgeChoices[draw][0], edgeChoices[draw][1]] = 1
						gene_plus[edgeChoices[draw][1], edgeChoices[draw][0]] = 1
					mutated_genes.append((gene[1], gene_plus))
				else:
					mutated_genes.append(gene)		
		else:	
			mutated_genes.append(gene)
	return mutated_genes

# Cross over genes
def motif_cross(population, probCross):
	cross_choice = []
	for i in range(len(population)-1):
		for j in range(i, len(population)):
			cross_choice.append((i, j))
	choice_index = range(len(cross_choice))
	draw = np.random.choice(choice_index, int(len(choice_index)*probCross), replace=False)
	result_population = population
	for item in draw:
		gene_ind = np.random.choice(range(3))
		temp = copy.copy(result_population[cross_choice[item][0]][gene_ind])
		result_population[cross_choice[item][0]][gene_ind] = result_population[cross_choice[item][1]][gene_ind]
		result_population[cross_choice[item][1]][gene_ind] = temp
	return result_population

# Eliminate the worst performing genes
def motif_select(candidateList, scoreList, numSurvivals):
	score_candidate_pair = zip(scoreList, candidateList)		
	score_candidate_pair = sorted(score_candidate_pair, reverse=True, key=lambda x:x[0])
	survived_candidates = [item[1] for item in score_candidate_pair[:numSurvivals]]
	survived_scores = [item[0] for item in score_candidate_pair[:numSurvivals]]
	return survived_candidates, survived_scores

# Repopulate the gene pool with the best performing genes
def motif_reproduce(candidateList, scoreList, numPopulation):
	if numPopulation > len(candidateList):
		for i in range(numPopulation - len(candidateList)):	
			candidateList.append(candidateList[i])
	return candidateList

# Generate the adj matrix corresponding to the given genes
def construct_motif_adj_batch(motifCandidates, adj_dic, baseADJ, flagd, flagacc):
	adjList = []
	numNodes = len(baseADJ)
	for candidate in motifCandidates:
		candidate_adj = []
		for gene in candidate:
			if str(list(gene[1].flatten())) in adj_dic:
				candidate_adj.append(adj_dic[str(list(gene[1].flatten()))])
			else:
				resultADJ = [np.eye(numNodes)] # self-loop
				print("Candidate motif: " + str(list(np.reshape(gene[1], -1)))+ ", with ancestor: ", str(list(np.reshape(gene[0], -1))))
				search.init_incsearch(gene[0], gene[1])
				print("  Start Inc searching...")
				while(1):
					result_temp = np.array(search.readout(numNodes*numNodes+1))
					resultADJ.append(np.reshape(result_temp[:-1], (numNodes, numNodes)))
					if result_temp[-1]==0:
						break

				resultADJ = [sparse_mx_to_torch_sparse_tensor(normalize(sp.csr_matrix(item))) for item in resultADJ]
				# resultADJ = [sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(item)) for item in resultADJ]
				candidate_adj.append(resultADJ)
				adj_dic[str(list(gene[1].flatten()))] = resultADJ
		adjList.append(candidate_adj)
	return adjList, adj_dic

# Generate unique representations of the given subgraphs
def motif_canonical(motif, flagd):
	motif_input = np.reshape(motif, -1)
	search.canonical(motif_input, flagd)
	return str(motif_input)

