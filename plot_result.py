import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
import pickle


TIME = '20181019121427'


#####################
# FP model
###################

#Calculate average
def cal_average_fp(result_list):
	#Convert generator to list
	input_list = list(result_list)
	avg_list = []
	#average
	for i in range(len(input_list)):
		avg_hit_rate = (np.mean(input_list[i][3]))*100
	#	print ('Average hit rate:', avg_hit_rate)
		avg_list.append(avg_hit_rate)	
	return avg_list



#Calculate standard diviation
def cal_deviation_fp(result_list):
	input_list = list(result_list)
	dev_list = []

	for i in range(len(input_list)):
		dev_hit_rate = (np.std(input_list[i][3], ddof=1))*5
	#	print ('Average hit rate:', avg_hit_rate)
		dev_list.append(dev_hit_rate)	
	return dev_list

#Retrive cache cost data
def retrive_cost_fp(result_list):
	input_list = list(result_list)
	cost_list = []

	for i in range(len(input_list)):
		cost_list.append(input_list[i][7])
	return cost_list


#################
# Base (Lazy) model
#################

#Calculate average
def cal_average_base(result_list):
	#Convert generator to list
	input_list = list(result_list)
	avg_list = []
	#average
	for i in range(len(input_list)):
		avg_hit_rate = (np.mean(input_list[i][2]))*100
	#	print ('Average hit rate:', avg_hit_rate)
		avg_list.append(avg_hit_rate)	
	return avg_list



#Calculate standard diviation
def cal_deviation_base(result_list):
	input_list = list(result_list)
	dev_list = []

	for i in range(len(input_list)):
		dev_hit_rate = (np.std(input_list[i][2], ddof=1))*5
	#	print ('Average hit rate:', avg_hit_rate)
		dev_list.append(dev_hit_rate)	
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

##############
# Main function
##############


file_name_fp = 'test_result/fp_raw_result_%s' %TIME
file_name_base = 'test_result/base_raw_result_%s' %TIME

result_list_fp = list(load_input(file_name_fp))
result_list_base = list(load_input(file_name_base))

avg_list_fp = cal_average_fp(result_list_fp)
dev_list_fp = cal_deviation_fp(result_list_fp)

avg_list_base = cal_average_base(result_list_base)
dev_list_base = cal_deviation_base(result_list_base)


'''print("Average:", avg_list)
print("Deviation:", dev_list)'''



#plot result
x = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4192]
xi = np.arange(0, len(x))

#y = np.array(avg_list)
#e = np.array(dev_list)

#plt.axis([0,len(avg_list), 20, 80])
plt.figure(1)
line1 = plt.errorbar(xi, avg_list_fp, dev_list_fp, linewidth=2, linestyle='-', color='green', ecolor='red',
 marker='o',mfc='blue', mec='blue', fmt='o', elinewidth=1, capsize=3,
 )
line2 = plt.errorbar(xi, avg_list_base, dev_list_base, linewidth=2, linestyle='-.', color='yellow', ecolor='red',
 marker='o',mfc='blue', mec='blue', fmt='o', elinewidth=1, capsize=3,
 )

plt.legend((line1, line2), ('FP-Growth model', 'Baseline/Lazy model'))
#plt.legend(handles=[line1])

plt.title('FP-Growth model vs. Lazy model average cache hit rate \n (Errorbar enlarge 5x)')

plt.xlabel('Cache size (number of elements)')
plt.ylabel('Cache hit rate %')
plt.xticks(xi,x)


plt.savefig('figures/cache_hit_rate_%s.png' %TIME)


plt.figure(2)
plt.title('Data fetch cost \n FP-Growth model vs. Lazy model ')

plt.xlabel('Cache size (number of elements)')
plt.ylabel('Cost')
plt.xticks(xi,x)
line3, = plt.plot(xi, retrive_cost_fp(result_list_fp), color='green', marker='o')
line4, = plt.plot(xi, retrive_cost_base(result_list_base), color='yellow', marker='o')
plt.legend((line3, line4), ('FP-Growth model', 'Baseline/Lazy model'))

plt.savefig('figures/cache_cost_%s.png' %TIME)
plt.show()

#print(retrive_cost_fp(result_list_fp))

