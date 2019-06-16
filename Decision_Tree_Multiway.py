import csv
import numpy as np

class DecisionTree():
	
	def learn(self, training_set):
        # implement this function
		
		def split_tree(attr,val1,val2,ipdata):
			tree_left = []
			tree_centre = []
			tree_right = []
			for ix in ipdata:
				if ix[attr] < val1:
					tree_left.append(ix)
				elif ix[attr] >= val1 and ix[attr] < val2:
					tree_centre.append(ix)
				else:
					tree_right.append(ix)
			return tree_left,tree_centre,tree_right
		
		def calc_entropy(ip_vec):
			ix0_count = 0
			ix1_count = 0
			for ix in ip_vec:
				if ix == 0:
					ix0_count += 1
				else:
					ix1_count += 1
			if ix0_count == 0 or ix1_count == 0:
				return 0
			else:
				return (-1*(((ix0_count/(ix0_count+ix1_count))*(np.log2((ix0_count/(ix0_count+ix1_count)))))+((ix1_count/(ix0_count+ix1_count))*(np.log2(ix1_count/(ix0_count+ix1_count))))))
			
		def get_ratio(child_tree,i):
			cnt0 = 0
			cnt1 = 0
			for ix in (ix[i] for ix in child_tree):
				if ix == 0:
					cnt0 += 1
				else:
					cnt1 += 1
			if cnt0 == 0 and cnt1 == 0:
				return (0,0)
			else:
				return(cnt0/(cnt0+cnt1),cnt1/(cnt0+cnt1))			
	
		def calc_info_gain(parent_ent,i,tree_lt,tree_ct,tree_rt):
			lt_entropy = calc_entropy([ix[-1] for ix in tree_lt])
			ct_entropy = calc_entropy([ix[-1] for ix in tree_ct])
			rt_entropy = calc_entropy([ix[-1] for ix in tree_rt])
			tmp_gain = parent_ent - ((len(tree_lt)/(len(tree_lt)+len(tree_ct)+len(tree_rt)))*lt_entropy) - ((len(tree_ct)/(len(tree_lt)+len(tree_ct)+len(tree_rt)))*ct_entropy) - ((len(tree_rt)/(len(tree_lt)+len(tree_ct)+len(tree_rt)))*rt_entropy)
			return tmp_gain
		
		def chk_last_node(ipSet):
			class_val = [elem[-1] for elem in ipSet]
			response = max(set(class_val),key=class_val.count)
			return(response)
		
		def get_range_val(ipdata,col):
			range_val = [float(elem[col]) for elem in ipdata]
			mean_val = sum(range_val)/len(range_val)
			first_val = (range_val[0] + mean_val)/2
			second_val = (range_val[-1] + mean_val)/2
			return(first_val,second_val)
		
		def get_child(ipdata):
		
			def sortByIndex(item):
				return item[3]
			
			decision_cols = list(range(len(ipdata[0])-1))
			parent_entropy = calc_entropy([ip[-1] for ip in ipdata])
			info_gain = []
			
			for ix in decision_cols:
				ix_val1,ix_val2 = get_range_val(ipdata,ix)
				result_left,result_centre,result_right = split_tree(ix,ix_val1,ix_val2,ipdata)
				info_gain.append((ix,ix_val1,ix_val2,calc_info_gain(parent_entropy,ix,result_left,result_centre,result_right),result_left,result_centre,result_right))
			
			info_gain=sorted(info_gain,key=sortByIndex,reverse=True)
			return {'attribute':info_gain[0][0],'chk_value1':info_gain[0][1],'chk_value2':info_gain[0][2],'left_branch':info_gain[0][4],'centre_branch':info_gain[0][5],'right_branch':info_gain[0][6]}
		
		def splitTree(main_tree):
			left_child = main_tree['left_branch']
			centre_child = main_tree['centre_branch']
			right_child = main_tree['right_branch']
			
			#print("Splitting into " + str(len(left_child)) + " , " + str(len(centre_child)) + " & " + str(len(right_child)))

			del(main_tree['left_branch'])
			del(main_tree['centre_branch'])
			del(main_tree['right_branch'])
			
			if not left_child and centre_child and right_child:
				main_tree['left_child'] = chk_last_node(centre_child+right_child)
				main_tree['right_child'] = main_tree['left_child']
				main_tree['centre_child'] = main_tree['left_child']
				return

			if not centre_child and left_child and right_child:
				main_tree['centre_child'] = chk_last_node(left_child+right_child)
				main_tree['left_child'] = main_tree['centre_child']
				main_tree['right_child'] = main_tree['centre_child']
				return

			if not right_child and left_child and centre_child:
				main_tree['right_child'] = chk_last_node(left_child+centre_child)
				main_tree['left_child'] = main_tree['right_child']
				main_tree['centre_child'] = main_tree['right_child']
				return

			if not left_child and not centre_child and right_child:
				main_tree['right_child'] = chk_last_node(right_child)
				main_tree['left_child'] = main_tree['right_child']
				main_tree['centre_child'] = main_tree['right_child']
				return

			if not centre_child and not right_child and left_child:
				main_tree['left_child'] = chk_last_node(left_child)
				main_tree['right_child'] = main_tree['left_child']
				main_tree['centre_child'] = main_tree['left_child']
				return

			if not left_child and not right_child and centre_child:
				main_tree['centre_child'] = chk_last_node(centre_child)
				main_tree['left_child'] = main_tree['centre_child']
				main_tree['right_child'] = main_tree['centre_child']
				return				
			
			if not left_child and not right_child and not centre_child:
				main_tree['left_child'] = main_tree['centre_child'] = main_tree['right_child'] = 0
				return

			main_tree['left_child'] = get_child(left_child)
			splitTree(main_tree['left_child'])
			
			main_tree['centre_child'] = get_child(centre_child)
			splitTree(main_tree['centre_child'])			
				
			main_tree['right_child'] = get_child(right_child)
			splitTree(main_tree['right_child'])
				
			return main_tree
		
		## Start of the functin
		self.tree = {}
		self.tree = get_child(training_set)
		splitTree(self.tree)
	

    # implement this function
	def classify(self, test_instance):
		result = 0 # baseline: always classifies as 0
		
		def get_class(tree_input,test_data):
			if test_data[tree_input['attribute']] < tree_input['chk_value1']:
				if isinstance(tree_input['left_child'], dict):
					return get_class(tree_input['left_child'],test_data)
				else:
					return tree_input['left_child']
			elif test_data[tree_input['attribute']] >= tree_input['chk_value1'] and test_data[tree_input['attribute']] < tree_input['chk_value2']:
				if isinstance(tree_input['centre_child'], dict):
					return get_class(tree_input['centre_child'],test_data)
				else:
					return tree_input['centre_child']
			else:
				if isinstance(tree_input['right_child'], dict):
					return get_class(tree_input['right_child'],test_data)
				else:
					return tree_input['right_child']
		
		result = get_class(self.tree,test_instance)
		return result


# Load data set
with open("wine-dataset.csv") as f:
    next(f, None)
    data = [tuple(line) for line in csv.reader(f, delimiter=",")]
print("Number of records: %d" % len(data))

def run_decision_tree():

	tree = DecisionTree()

	# Split training/test sets
	# You need to modify the following code for cross validation.
	K = 10

	accuracy_list = []
	for ix in list(range(10)):
		training_set = [tuple(map(float,x)) for i, x in enumerate(data) if i % K != ix]
		test_set = [tuple(map(float,x)) for i, x in enumerate(data) if i % K == ix]

		# Construct a tree using training set, set maximum depth to prevent looping
		output = tree.learn(training_set)

		# Classify the test set using the tree we just constructed
		results = []
		for instance in test_set:
			result = tree.classify( instance[:-1] )
			results.append( result == instance[-1])

		# Accuracy
		accuracy = float(results.count(True))/float(len(results))
		
		# Create a list of accuracy scores for each fold
		accuracy_list.append(accuracy)

	accuracy  = sum(accuracy_list)/float(len(accuracy_list))
	print("Mean Accuracy using 10 fold validation (Multiway Decision Tree): " + str(accuracy))


	# Writing results to a file
	f = open(myname+"result.txt", "w")
	f.write("accuracy: %.4f" % accuracy)
	f.close()


if __name__ == "__main__":
    run_decision_tree()