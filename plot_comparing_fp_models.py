import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
import pickle


TIME = '20181019172440'


#####################
# FP model
###################

#Calculate average
def cal_average_fp(result_list):
	#Convert generator to list
	input_list = list(result_list)
	avg_list = []
	avg_list_tmp = []
	#average
	#print("input_list:", input_list)

	for i in range(len(input_list)):
		avg_hit_rate = (np.mean(input_list[i][3]))*100
	#	print ('Average hit rate:', avg_hit_rate)
		avg_list_tmp.append(avg_hit_rate)	
		#print('avg_list_tmp', avg_list_tmp)
		if (i+1)%11 == 0:
			avg_list.append(avg_list_tmp)
			avg_list_tmp = []

	return avg_list



#Calculate standard diviation
def cal_deviation_fp(result_list):
	input_list = list(result_list)
	dev_list = []
	dev_list_tmp = []

	for i in range(len(input_list)):
		dev_hit_rate = (np.std(input_list[i][3], ddof=1))*5
	#	print ('Average hit rate:', avg_hit_rate)
		dev_list_tmp.append(dev_hit_rate)	
		if (i+1)%11 == 0:
			dev_list.append(dev_list_tmp)
			dev_list_tmp = []
	
	return dev_list

#Retrive cache cost data
def retrive_cost_fp(result_list):
	input_list = list(result_list)
	cost_list = []
	cost_list_tmp = []

	for i in range(len(input_list)):
		cost_list_tmp.append(input_list[i][7])
		if (i+1)%11 == 0:
			cost_list.append(cost_list_tmp)
			cost_list_tmp = []
	return cost_list

#Retrive cache hit data
def retrive_hit_fp(result_list):
	input_list = list(result_list)
	hit_list = []

	for i in range(len(input_list)):
		hit_list.append(input_list[i][4])
	return hit_list

#Retrive cache miss data
def retrive_miss_fp(result_list):
	input_list = list(result_list)
	miss_list = []

	for i in range(len(input_list)):
		miss_list.append(input_list[i][5])
	return miss_list

#Retrive cache prefetch data
def retrive_prefetch_fp(result_list):
	input_list = list(result_list)
	prefetch_list = []

	for i in range(len(input_list)):
		prefetch_list.append(input_list[i][6])
	return prefetch_list

#################
# Base (Lazy) model
#################

#Calculate average
def cal_average_base(result_list):
	#Convert generator to list
	input_list = list(result_list)
	avg_list = []
	avg_list_tmp = []
	#average
	#print("input_list:", input_list)

	for i in range(len(input_list)):
		avg_hit_rate = (np.mean(input_list[i][2]))*100
	#	print ('Average hit rate:', avg_hit_rate)
		avg_list_tmp.append(avg_hit_rate)	
		#print('avg_list_tmp', avg_list_tmp)
		if (i+1)%11 == 0:
			avg_list.append(avg_list_tmp)
			avg_list_tmp = []

	return avg_list



#Calculate standard diviation
def cal_deviation_base(result_list):
	input_list = list(result_list)
	dev_list = []
	dev_list_tmp = []

	for i in range(len(input_list)):
		dev_hit_rate = (np.std(input_list[i][2], ddof=1))
	#	print ('Average hit rate:', avg_hit_rate)
		dev_list_tmp.append(dev_hit_rate)	
		if (i+1)%11 == 0:
			dev_list.append(dev_list_tmp)
			dev_list_tmp = []
	
	return dev_list


#Load input file
def load_input(file_name):
	with open (file_name, 'rb') as file:
		input_file = pickle.load(file)
	return input_file

#Retrive cache cost data
def retrive_cost_base(result_list):
	input_list = list(result_list)
	cost_list = []
	for i in range(len(input_list)):
		cost_list.append(input_list[i][5])
	return cost_list

#Retrive cache hit data
def retrive_hit_base(result_list):
	input_list = list(result_list)
	hit_list = []

	for i in range(len(input_list)):
		hit_list.append(input_list[i][3])
	return hit_list

#Retrive cache miss data
def retrive_miss_base(result_list):
	input_list = list(result_list)
	miss_list = []

	for i in range(len(input_list)):
		miss_list.append(input_list[i][4])
	return miss_list

##############
# Main function
##############


file_name_fp = 'test_result/fp_raw_result_%s' %TIME

result_list_fp = list(load_input(file_name_fp))

avg_list_fp = cal_average_fp(result_list_fp)
dev_list_fp = cal_deviation_fp(result_list_fp)

#print (avg_list_fp[0])


'''print("Average:", avg_list)
print("Deviation:", dev_list)'''



#plot result
x = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4192]
xi = np.arange(0, len(x))


#################
# Cache hit rate
#################

plt.figure(1)
line1 = plt.errorbar(xi, avg_list_fp[0], dev_list_fp[0], linewidth=2, linestyle='-', color='green', ecolor='red',
 marker='o',mfc='blue', mec='blue', fmt='o', elinewidth=1, capsize=3)
line2 = plt.errorbar(xi, avg_list_fp[1], dev_list_fp[1], linewidth=2, linestyle='-', color='yellow', ecolor='red',
 marker='o',mfc='blue', mec='blue', fmt='o', elinewidth=1, capsize=3)
line3 = plt.errorbar(xi, avg_list_fp[2], dev_list_fp[2], linewidth=2, linestyle='-', color='cyan', ecolor='red',
 marker='o',mfc='blue', mec='blue', fmt='o', elinewidth=1, capsize=3)
line4 = plt.errorbar(xi, avg_list_fp[3], dev_list_fp[3], linewidth=2, linestyle='-', color='magenta', ecolor='red',
 marker='o',mfc='blue', mec='blue', fmt='o', elinewidth=1, capsize=3)


plt.legend((line1, line2, line3, line4), ('FP-Growth support: 4000', 'FP-Growth support: 6000', 'FP-Growth support: 8000', 'FP-Growth support: 10000'))
#plt.legend(handles=[line1])

plt.title('Comparing cache hit rate with different FP-Growth support numbers \n (Errorbar enlarge 5x)')

plt.xlabel('Cache size (number of elements)')
plt.ylabel('Cache hit rate %')
plt.xticks(xi,x)


plt.savefig('figures/FP_support_cache_hit_rate_%s.png' %TIME)


#################
# CData fetch cost
#################

plt.figure(2)
plt.title('Data fetch cost \n FP-Growth model vs. Lazy model ')

plt.xlabel('Cache size (number of elements)')
plt.ylabel('Cost')
plt.xticks(xi,x)

cost_list = retrive_cost_fp(result_list_fp)

line1, = plt.plot(xi, cost_list[0], color='green', marker='o')
line2, = plt.plot(xi, cost_list[1], color='yellow', marker='o')
line3, = plt.plot(xi, cost_list[2], color='cyan', marker='o')
line4, = plt.plot(xi, cost_list[3], color='magenta', marker='o')

line5 = plt.hlines(y=1148*100, xmin=0, xmax=10, color="red")
plt.legend((line1, line2, line3, line4, line5), ('FP-Growth support: 4000', 'FP-Growth support: 6000', 'FP-Growth support: 8000', 'FP-Growth support: 10000', 'Naive'))
#plt.legend((line1, line2), ('FP-Growth model', 'Baseline/Lazy model'))

plt.savefig('figures/FP_support_cache_cost_%s.png' %TIME)


'''#################
# Total hit, miss and pretech
#################


plt.figure(3)
plt.title('Total number of cache hit/miss/prefetch \n FP-Growth model vs. Lazy model ')

plt.xlabel('Cache size (number of elements)')
plt.ylabel('Total numbers')
plt.xticks(xi,x)
line1, = plt.plot(xi, retrive_hit_fp(result_list_fp), color='green', marker='o', linestyle='-', mfc='red', mec='red')
line2, = plt.plot(xi, retrive_hit_base(result_list_base), color='yellow', marker='o', linestyle='-', mfc='red', mec='red')

line3, = plt.plot(xi, retrive_miss_fp(result_list_fp), color='green', marker='o', linestyle='-.', mfc='blue', mec='blue')
line4, = plt.plot(xi, retrive_miss_base(result_list_base), color='yellow', marker='o', linestyle='-.', mfc='blue', mec='blue')

line5, = plt.plot(xi, retrive_prefetch_fp(result_list_fp), color='green', marker='o', linestyle='--', mfc='black', mec='black')

plt.legend((line1, line2, line3, line4, line5), ('FP cache hit', 'Baseline cache hit', 'FP cache miss', 'Baseline cache miss', 'FP prefetch'))

plt.savefig('figures/cache_hit_miss_prefetch_%s.png' %TIME)'''



plt.show()































