import fp_growth_py3 as fpg
import pickle

dataset = [line.split() for line in open('kosarak.dat').readlines()]


for i in [3000, 2000]:

	fp_model = fpg.find_frequent_itemsets(dataset, minimum_support=i, include_support=True)

	fp_model_list = []
	for itemset, support in fp_model:    # 将generator结果存入list
		fp_model_list.append((itemset, support))

	with open('fp_model_support_%d' %i, 'wb') as fp:
		pickle.dump(fp_model_list, fp)

'''with open ('fp_modle_support_10000', 'rb') as fp:
	fp_model = pickle.load(fp)


print (fp_model)'''