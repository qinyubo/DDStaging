import fp_growth_py3 as fpg
import pickle
import random

dataset = [line.split() for line in open('kosarak.dat').readlines()]


def find_input(dataset, num_input):
	input_list = []
	#arndomly sample num_input from dataset without duplications
	input_list = random.sample(dataset, num_input)

	return input_list

#for i in range(100, 1000, 100):
for i in [1000]:
	input_list = []
	input_list = find_input(dataset, i)

	with open('input_dataset/input_%d' %i, 'wb') as inputs:
		pickle.dump(input_list, inputs)
