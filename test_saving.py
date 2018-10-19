import pickle

a = [1,3,4,5,6]

m='hell'
n='sdfsdf'

with open('%sfp_model_support_%s' %(m,n), 'wb') as fp:
	pickle.dump(a, fp)

'''with open ('fp_modle_support_10000', 'rb') as fp:
	fp_model = pickle.load(fp)


print (fp_model)'''