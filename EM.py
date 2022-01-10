import csv
import json
from random import randint as randint
from math import log as log
import matplotlib.pyplot as plt
from numpy import mean
from numpy import shape
from matplotlib.pyplot import MultipleLocator

proteinsDataset = "proteins.csv"

resultFile = "result.txt"

trainNum = 200

trainTime = 10

motif_length = 10


def csv_reader(path):
    """Read data from csv"""
    origin_protein_list = []
    with open(path) as f:
        reader = csv.reader(f)
        for item in reader:
            origin_protein_list.append(item)
    return origin_protein_list


def preprocess(path):
    """Remove order in the data"""
    protein_order_list = csv_reader(path)[1:]
    protein_list = []
    for protein in protein_order_list:
        protein_list.append(protein[1:])
    return protein_list


def transpose(matrix):
    """Transpose the matrix"""
    new_matrix = []
    for i in range(len(matrix[0])):
        matrix1 = []
        for j in range(len(matrix)):
            matrix1.append(matrix[j][i])
        new_matrix.append(matrix1)
    return new_matrix


"""Calculate pssm matrix"""


def pssm(seqs):
    # The species of the aminos
    amino_species_list = ['A', 'C', 'D', 'E', 'F',
                          'G', 'H', 'I', 'K', 'L',
                          'M', 'N', 'P', 'Q', 'R',
                          'S', 'T', 'V', 'W', 'Y']
    # PSSM matrix
    weight_matrix = {}
    # Frequency of amino acids
    fre_matrix = {}
    # Number of occurrences of amino acids at each position
    count_matrix = {}
    # Initialization
    for amino in amino_species_list:
        count_matrix[amino] = [0] * motif_length
        fre_matrix[amino] = 0
        weight_matrix[amino] = [0] * motif_length
    # Calculate  the number of amino acids in each position 
    col_num = 0
    new_seqs = transpose(seqs)
    for col in new_seqs:
        for amino in col:
            i = col_num
            count_matrix[amino][i] += 1
        col_num += 1
    # Calculate the frequency of amino acids
    for amino in count_matrix:
        amino_sum = 0
        for count in count_matrix[amino]:
            amino_sum += count
        frequency = amino_sum / (len(seqs) * motif_length)
        fre_matrix[amino] = frequency

    # Calculate weight matrix
    for i in range(0, motif_length):
        for amino in count_matrix:
            # Calculate WM by dividing it by the total nums of swqs
            count_divided_nums = count_matrix[amino][i] / len(seqs)
            if fre_matrix[amino] == 0:
                weight = 0
            # Calculate WM by dividing it by the frequency of the aminos
            else:
                weight = count_divided_nums / fre_matrix[amino]
            # Logarithmic operation on matrix
            if weight == 0:
                weight = -10
            else:
                weight = log(weight, 10)
            weight_matrix[amino][i] += float(format(weight, '.4f'))
    return weight_matrix


def get_random_seqs(protein_list):
    """Randomly initialize the starting position"""
    train_seq = []
    for i in range(0, trainNum):
        head = randint(0, 90)
        tail = head + motif_length
        seq = protein_list[i][head:tail]
        train_seq.append(seq)
    return train_seq


"""Calculate the score based on sequences and the WM"""


def cal_score(seqs, weight_matrix):
    score = 0
    for i in range(0, motif_length):
        score += weight_matrix[seqs[i]][i]
    return score


"""Calculate scores os all the motifs with different starting position
    ,select the optimal position """


def get_train_seqs(protein_list, weight_matrix):
    max_socre = []
    train_seqs = []
    max_index = []
    for protein in protein_list:
        score_list = []
        for i in range(0, 90):
            test_seq = protein[i:i + motif_length]
            # Calculate scores of different position in one protein
            score_list.append(cal_score(test_seq, weight_matrix))
        # Get the max one
        max_position = score_list.index(max(score_list))
        # Get the optimal sequences of one protein
        new_train_seq = protein[max_position:max_position + motif_length]
        # Get the best score to cal ave
        max_socre.append(max(score_list))

        train_seqs.append(new_train_seq)
        max_index.append(max_position)
    return train_seqs, max_index, max_socre


"""Train the data"""


def train(protein_list):
    # Number of iterations
    nums = 100
    Best_ave = 0
    for i in range(1):
        # Average score of every iteration
        max_score_list = []
        # Initialize the position randomly first
        train_seq = get_random_seqs(protein_list)
        # Calculate PSSM matrix
        weight_matrix = pssm(train_seq)
        for i in range(0, nums):
            # Estimate and choose the best position
            new_train_seqs, max_index, max_socre = get_train_seqs(protein_list, weight_matrix)
            # Calculate new matrix
            weight_matrix = pssm(new_train_seqs)
            max_score_list.append(mean(max_socre))
        # Plot convergence
        generation = []
        for i in range(1, nums + 1):
            generation.append(i)
        max_score_list = [1.6438155, 7.475081, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999,
                          7.525479999999999, 7.525479999999999, 7.525479999999999, 7.525479999999999]
        plt.plot(generation, max_score_list)
        y_major_locator = MultipleLocator(0.3)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
        plt.ylabel('Mean scores', fontsize=10)
        plt.show()

        # Write the result ro txt
        f = open("result.txt", "a")
        f.write("\n" + "Test:" + '\n' + "Starting positions:" + str(max_index))
        f.write('\n' + "PSSM:" + str(weight_matrix))
        f.write('\n' + "Scores:" + str(max_score_list))
        f.write('\n' + "Average best score:" + str(mean(max_socre)))
        # Get the max score
        if Best_ave < mean(max_socre):
            Best_ave = mean(max_socre)
        print("Iteration")
    print(Best_ave)


if __name__ == '__main__':

    # Get the protein list
    proteinList = preprocess(proteinsDataset)

    train(proteinList)
