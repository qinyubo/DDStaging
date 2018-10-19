import fp_growth_py3 as fp
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import csv

#Data set
dataset = [['r', 'z', 'h', 'j', 'p'],
			   ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
			   ['z'],
			   ['r', 'x', 'n', 'o', 's'],
			   ['y', 'r', 'x', 'z', 'q', 't', 'p'],
			   ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]

dataset = [line.split() for line in open('kosarak.dat').readlines()]			   

#Define function
#FP, return generator
def fp_process(dataset, mini_sup, include_sup):
	frequent_itemsets = fp.find_frequent_itemsets(dataset, minimum_support=mini_sup, include_support=include_sup)
	return frequent_itemsets

#Process result, return list
def process_result(frequent_itemsets):
	result = []
	for itemset, support in frequent_itemsets:
		result.append((itemset, support))

	#result = sorted(result, key=lambda i: i[0])  

	return result

#Find matched element in result, return matches generator
#The element is the leaf of FP-tree, which is the last element in the matched_item
def find_matched_elem(result, element):
	matches = [matched_item for matched_item in result if matched_item[0][len(matched_item[0])-1] == element]
	return matches

#Find N most frequently successors, retune list
def find_succ(matches, num_succ):
	#sort matches base on its support in descent order
	#Sort based on length of matched itemsets
	matches = sorted(matches, key=lambda x:len(x[0]), reverse = True)

	matched_succssors = []

	for itemset, support in matches:
		for elem in reversed(itemset):
			if elem not in matched_succssors:
				if len(matched_succssors) < num_succ:
					matched_succssors.append(elem)
				else:
					break

	return matched_succssors

#Find random input sets from datasets, return a list
def find_input(dataset, num_input):
	input_list = []
	#arndomly sample num_input from dataset without duplications
	input_list = random.sample(dataset, num_input)

	return input_list

#Load pre-proccessed fp-model
def load_fp_model(file_name):
	with open (file_name, 'rb') as fp:
		fp_model = pickle.load(fp)
	return fp_model

#Load pre-selected input data set
def load_input_dataset(file_name):
	with open (file_name, 'rb') as inputs:
		inputs = pickle.load(inputs)
	return inputs

#Match input and cache
def evaluate_cache(dataset, input_set, fp_model, cache_size):
	size_single_input = 0 #size of single input set
	num_hit = 0 #number of cache hit for a single input set
	num_miss = 0 # number of cache miss for a single input set
	hit_rate = 0 #hit rate for a single input set
	hit_rate_list = [] #list of hit rate for all input data set
	all_hit_rate_list = []
	iterate = 0; #total iteration of experiment
	num_inputs = 100; #number of input sets that randomly fetch from dataset
	num_elem_cache = cache_size #number of element fetched to cache

	#process data
	#freq_itemsets = fp_process(dataset, 2, True)

	#load pre-processed fp models
	freq_itemsets = load_fp_model(fp_model)


	#process result
	processed_fp = []
	processed_fp = process_result(freq_itemsets)

	'''	#Main iteration, design for random inputs
	for i in range(tot_iter):
		#generate random inputset
		#input_set = find_input(dataset, num_inputs)

		#load input data
		input_set = load_input_dataset('input_%d' %num_inputs)

		#print ('input_set:', input_set)
		hit_rate_list = []

		for itemset in input_set:
			#init, find the matched element for the first element in the itemset
			matches = find_matched_elem(processed_fp, itemset[0])
			cache_elem = find_succ(matches, num_elem_cache)
			#print ('cache_elem1:',cache_elem)
			size_single_input = len(itemset)
			hit_rate = 0
			num_hit = 0
			num_miss = 0
			
			#iterate itemset
			for elem in itemset:
				if elem in cache_elem:
					num_hit += 1
					#print ('HIT:', 'itemset:', itemset, 'elem:', elem, 'cache:', cache_elem)
					#continue
				else:
					num_miss += 1
					#print ('MISS:', 'itemset:', itemset, 'elem:', elem, 'cache:', cache_elem)
					#refetch the cache elements based on this missed element
					matches = find_matched_elem(processed_fp, elem)
					new_cache_elem = find_succ(matches, num_elem_cache)
					#merge new_cache_elem with existed one without duplications
					cache_elem = list(set(new_cache_elem + cache_elem))
			hit_rate = num_hit / size_single_input
			hit_rate_list.append(hit_rate)

			#print ('cache_elem2:',cache_elem)
			#print ('num_hit:', num_hit, 'size_single_input:', size_single_input, 'hit rate:', num_hit / size_single_input) 
		
		#store all hit rate result for all iteration	
		all_hit_rate_list.append(hit_rate_list)

	return all_hit_rate_list'''
	

	#Main loop for loading input
	#load input data
	input_set = load_input_dataset(input_set)

	#print ('input_set:', input_set)
	hit_rate_list = []

	for itemset in input_set:
		#init, find the matched element for the first element in the itemset
		matches = find_matched_elem(processed_fp, itemset[0])
		cache_elem = find_succ(matches, num_elem_cache)
		#print ('cache_elem1:',cache_elem)
		size_single_input = len(itemset)
		hit_rate = 0
		num_hit = 0
		num_miss = 0
		
		#iterate itemset
		for elem in itemset:
			if elem in cache_elem:
				num_hit += 1
				#continue
			else:
				num_miss += 1
				#refetch the cache elements based on this missed element
				matches = find_matched_elem(processed_fp, elem)
				new_cache_elem = find_succ(matches, num_elem_cache)
				
				#evict cache when it is full
				if len(cache_elem) >= num_elem_cache:
					#randomly pop out len(new_cache_elem) element
					for i in range(len(new_cache_elem)):
						cache_elem.pop(random.randrange(len(cache_elem)))
				#merge new_cache_elem with existed one without duplications
				cache_elem = list(set(new_cache_elem + cache_elem))

		hit_rate = num_hit / size_single_input
		hit_rate_list.append(hit_rate)

	return hit_rate_list


#Process the result and generate statistic data, for loading random input
def process_result_list_random_input(result_list):
	#Convert generator to list
	res_list = list(result_list)
	#average
	avg_list = []

	for i in range (len(res_list)):
		avg_list.append(np.mean(res_list[i]))
		#print ('average:', np.mean(res_list[i]))
	
	print (avg_list)
	print ('Average hit rate:', np.mean(avg_list))

	'''	plt.plot(avg_list)
	plt.title('Average hit rate')
	plt.xlim([1, len(avg_list)])
	plt.show()'''
	return None

#For processing loaded input
def process_result_list_loaded_input(result_list):
	#Convert generator to list
	input_list = list(result_list)
	avg_list = []
	#average
	for i in range(len(input_list)):
		avg_hit_rate = np.mean(input_list[i][3])
	#	print ('Average hit rate:', avg_hit_rate)
		avg_list.append(avg_hit_rate)

	print('Avg hit rate:', avg_list)
	'''	
	plt.plot(res_list)
	plt.title('Hit rate')
	plt.xlim([1, len(res_list)])
	plt.show()'''
	return avg_list

#Save result list
def save_result_data(result_list, result_statistic_data):
	DIR = 'test_result/'
	TIME = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

	with open('%sraw_result_%s' %(DIR,TIME), 'wb') as raw_data:
		pickle.dump(result_list, raw_data)


	#Save statistic data to csv file
	with open('%sraw_result_%s.csv' %(DIR,TIME), 'w', newline='') as stat_data:
		wr = csv.writer(stat_data, quoting=csv.QUOTE_ALL)
		wr.writerow(result_statistic_data)

#Call fp module and execute FP-growth alg
if __name__ == '__main__':

	result_list = []
	result_statistic_data = []

	for inputs_size in [100,500,1000]:
		for fp_model_support_size in [3000, 6000, 9000]:
			for cache_size in [10,50,100]:
					result_list.append((
						inputs_size,
						fp_model_support_size,
						cache_size, 
						evaluate_cache(dataset, 
							'input_dataset/input_%d' %inputs_size,
							'fp_model/fp_model_support_%d' %fp_model_support_size,
							cache_size)
						))

	result_statistic_data = process_result_list_loaded_input(result_list)

	save_result_data(result_list, result_statistic_data)









