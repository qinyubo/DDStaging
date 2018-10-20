import fp_growth_py3 as fp
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import csv

#Global variables
TIME = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
TOT_HIT = 0
TOT_MISS =0
TOT_PREFETCH = 0

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

	#input_set_info(inputs)
	return inputs

#Calculate the total input elements
def input_set_info(input_file):
	input_elem_tmp = [] # temporily store input element for outputing input information

	for itemset in input_file:
		for elem in itemset:
			input_elem_tmp.append(elem)

	print("Total element:", len(input_elem_tmp))
	#find total unique elements
	print ('Total unique elements:', len(list(set(input_elem_tmp))))


#Evaluate cache with FP algorithm
def evaluate_cache_fp(dataset, input_set, fp_model, cache_size):
	size_single_input = 0 #size of single input set
	num_hit = 0 #number of cache hit for a single input set
	num_miss = 0 # number of cache miss for a single input set
	hit_rate = 0 #hit rate for a single input set
	hit_rate_list = [] #list of hit rate for all input data set
	all_hit_rate_list = []
	iterate = 0; #total iteration of experiment
	#num_inputs = 100; #number of input sets that randomly fetch from dataset
	num_elem_cache = 1 #number of element fetched to cache

	

	#Declare they are global variables
	global TOT_HIT
	global TOT_MISS
	global TOT_PREFETCH

	#Reset value
	TOT_HIT = 0
	TOT_MISS =0
	TOT_PREFETCH = 0

	#process data
	#freq_itemsets = fp_process(dataset, 2, True)

	#load pre-processed fp models
	freq_itemsets = load_fp_model(fp_model)


	#process result
	processed_fp = []
	processed_fp = process_result(freq_itemsets)
	

	#Main loop for loading input
	#load input data
	input_set = load_input_dataset(input_set)

	#print ('input_set:', input_set)
	hit_rate_list = []

	cache_elem = []

	for itemset in input_set:
		#init, find the matched element for the first element in the itemset
		matches = find_matched_elem(processed_fp, itemset[0])
		tmp_cache = find_succ(matches, num_elem_cache)
		cache_elem = list(set(cache_elem + tmp_cache))
		#print ('cache_elem1:',cache_elem)
		size_single_input = len(itemset)
		hit_rate = 0
		num_hit = 0
		num_miss = 0
		
		#iterate itemset
		for elem in itemset:
			if elem in cache_elem:
				num_hit += 1
				TOT_HIT += 1
				#continue
			else:
				num_miss += 1
				TOT_MISS += 1
				#refetch the cache elements based on this missed element
				matches = find_matched_elem(processed_fp, elem)
				new_cache_elem = find_succ(matches, num_elem_cache)

				#calculate the number of successors that need to be prefetch, which means not in the cache
				#I can calc the intersection of new_cache_elem with cache_elem
				TOT_PREFETCH += len(new_cache_elem) - len(set(new_cache_elem) & set(cache_elem))
				
				#evict cache when it is full
				if len(cache_elem) >= cache_size:
					#randomly pop out len(new_cache_elem) element
					for i in range(len(new_cache_elem)):
						cache_elem.pop(random.randrange(len(cache_elem)))
				#merge new_cache_elem with existed one without duplications
				cache_elem = list(set(new_cache_elem + cache_elem))
		
		hit_rate = num_hit / size_single_input
		hit_rate_list.append(hit_rate)
	

	#print ('TOT_HIT',TOT_HIT, 'TOT_MISS',TOT_MISS, 'TOT_PREFETCH',TOT_PREFETCH)

	return hit_rate_list



#Evaluate cache with passive strategy
def evaluate_cache_base(input_set, cache_size):
	size_single_input = 0 #size of single input set
	num_hit = 0 #number of cache hit for a single input set
	num_miss = 0 # number of cache miss for a single input set
	hit_rate = 0 #hit rate for a single input set
	hit_rate_list = [] #list of hit rate for all input data set
	all_hit_rate_list = []
	iterate = 0; #total iteration of experiment
	#num_inputs = 100; #number of input sets that randomly fetch from dataset


	#Declare they are global variables
	global TOT_HIT
	global TOT_MISS
	global TOT_PREFETCH

	#Reset value
	TOT_HIT = 0
	TOT_MISS =0
	TOT_PREFETCH = 0



	#Main loop for loading input
	#load input data
	input_set = load_input_dataset(input_set)

	#print ('input_set:', input_set)
	hit_rate_list = []

	cache_elem = []

	for itemset in input_set:
		size_single_input = len(itemset)
		hit_rate = 0
		num_hit = 0
		num_miss = 0
		
		#iterate itemset
		for elem in itemset:
			if elem in cache_elem:
				num_hit += 1
				TOT_HIT += 1
				#continue
			else:
				num_miss += 1
				TOT_MISS += 1
				#evict cache when it is full
				if len(cache_elem) >= cache_size:
					cache_elem.pop(random.randrange(len(cache_elem))) #randomly pop out one element
				else:
					#Put this element into cache
					cache_elem.append(elem)
				
		hit_rate = num_hit / size_single_input
		hit_rate_list.append(hit_rate)

	#print ('TOT_HIT',TOT_HIT, 'TOT_MISS',TOT_MISS, 'TOT_PREFETCH',TOT_PREFETCH)

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

#Calculate average
def cal_average(result_list):
	#Convert generator to list
	input_list = list(result_list)
	avg_list = []
	#average
	for i in range(len(input_list)):
		avg_hit_rate = np.mean(input_list[i][3])
	#	print ('Average hit rate:', avg_hit_rate)
		avg_list.append(avg_hit_rate)	
	return avg_list

#Calculate the total data fetch cost
#WARNING! it use global variables, be awared!
def cal_tot_cost():
	tot_cost = TOT_HIT*1 + TOT_MISS*120 + TOT_PREFETCH*50
	#print ('TOT_HIT',TOT_HIT, 'TOT_MISS',TOT_MISS, 'TOT_PREFETCH',TOT_PREFETCH, 'TOT_COST', tot_cost)
	return tot_cost

#Calculate standard diviation
def cal_deviation(result_list):
	input_list = list(result_list)
	dev_list = []

	for i in range(len(input_list)):
		dev_hit_rate = np.std(input_list[i][3], ddof=1)
	#	print ('Average hit rate:', avg_hit_rate)
		dev_list.append(dev_hit_rate)	
	return dev_list

#Save result list
def save_fp_raw_data(result_list):
	DIR = 'test_result/'

	with open('%sfp_raw_result_%s' %(DIR,TIME), 'wb') as raw_data:
		pickle.dump(result_list, raw_data)


#Save result list
def save_base_raw_data(result_list):
	DIR = 'test_result/'

	with open('%sbase_raw_result_%s' %(DIR,TIME), 'wb') as raw_data:
		pickle.dump(result_list, raw_data)

#Save statistic data to csv file
def save_statistic_date(file_1, file_2):
	DIR = 'test_result/'

	with open('%savg_result_%s.csv' %(DIR,TIME), 'w', newline='') as stat_data:
		wr = csv.writer(stat_data, quoting=csv.QUOTE_ALL)
		wr.writerow(file_1)

	with open('%sdev_result_%s.csv' %(DIR,TIME), 'w', newline='') as stat_data:
		wr = csv.writer(stat_data, quoting=csv.QUOTE_ALL)
		wr.writerow(file_2)

#Call fp module and execute FP-growth alg
if __name__ == '__main__':

	result_list_fp = []
	result_list_base = []
	#result_statistic_data = []
	result_avg = []
	result_dev = []


#Evaluate FP cache approach
	for inputs_size in [200]:
		for fp_model_support_size in [4000, 6000, 8000, 10000]:
			for cache_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4192]:
					result_list_fp.append((
						inputs_size,
						fp_model_support_size,
						cache_size, 
						evaluate_cache_fp(dataset, 
							'input_dataset/input_%d' %inputs_size,
							'fp_model/fp_model_support_%d' %fp_model_support_size,
							cache_size),
						TOT_HIT,
						TOT_MISS,
						TOT_PREFETCH,
						cal_tot_cost() #This function involve global variable, be awared!
						))
					print(inputs_size, fp_model_support_size, cache_size, "tot_cost:", cal_tot_cost())

	save_fp_raw_data(result_list_fp)
	'''	
	#result_statistic_data = process_result_list_loaded_input(result_list)
	result_avg = cal_average(result_list)
	result_dev = cal_deviation(result_dev)
	'''
	
	#save_statistic_date(result_avg, result_dev)


#Evaluate base cache approach
	for inputs_size in [200]:
		for cache_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4192]:
				result_list_base.append((
					inputs_size,
					cache_size, 
					evaluate_cache_base( 
						'input_dataset/input_%d' %inputs_size,
						cache_size),
					TOT_HIT,
					TOT_MISS,
					cal_tot_cost() #This function involve global variable, be awared!
					))
				print(inputs_size, cache_size, "tot_cost:", cal_tot_cost())

	save_base_raw_data(result_list_base)


	print ("Time:", TIME)







