import csv
import numpy as np

class DecisionTree():
	
	def learn(self, training_set):
		
		def split_tree(attr,val,ipdata):
			tree_left = []
			tree_right = []
			for ix in ipdata:
				if ix[attr] < val:
					tree_left.append(ix)
				else:
					tree_right.append(ix)
			return tree_left,tree_right
		
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
			
		def get_ratio(child_tree):
			cnt0 = 0
			cnt1 = 0
			for ix in child_tree:
				if ix == 0:
					cnt0 += 1
				else:
					cnt1 += 1
			if cnt0 == 0 and cnt1 == 0:
				return(0.5)
			return(1-((cnt0/(cnt0+cnt1))**2)-((cnt1/(cnt0+cnt1))**2))
	
		
		def chk_last_node(ipSet):
			class_val = [elem[-1] for elem in ipSet]
			response = max(set(class_val),key=class_val.count)
			return(response)
			
		def calc_gini_score(parent_gini,tree_lt,tree_rt):
			lt_gini = get_ratio([ix[-1] for ix in tree_lt])
			rt_gini = get_ratio([ix[-1] for ix in tree_rt])
			tmp_gini = parent_gini - ((len(tree_lt)/(len(tree_lt)+len(tree_rt)))*lt_gini) - ((len(tree_rt)/(len(tree_lt)+len(tree_rt)))*rt_gini)
			return tmp_gini			
			
		
		def get_child(ipdata):
		
			def sortByIndex(item):
				return item[2]
				
			decision_cols = list(range(len(ipdata[0])-1))
			parent_gini = get_ratio([ip[-1] for ip in ipdata])
			gini_gain = []
			
			for ix in decision_cols:
				ix_values = set(ix1[ix] for  ix1 in ipdata)
				for ix_val in ix_values:
					result_left, result_right = split_tree(ix,ix_val,ipdata)
					gini_gain.append((ix,ix_val,calc_gini_score(parent_gini,result_left,result_right),result_left,result_right))
			
			gini_gain=sorted(gini_gain,key=sortByIndex,reverse=True)
			return {'attribute':gini_gain[0][0],'chk_value':gini_gain[0][1],'left_branch':gini_gain[0][3],'right_branch':gini_gain[0][4]}
		
		def splitTree(main_tree):
			left_child = main_tree['left_branch']
			right_child = main_tree['right_branch']

			del(main_tree['left_branch'])
			del(main_tree['right_branch'])
			
			if not left_child and right_child:
				main_tree['left_child'] = chk_last_node(right_child)
				main_tree['right_child'] = chk_last_node(right_child)
				return
			
			if not right_child and left_child:
				main_tree['left_child'] = chk_last_node(left_child)
				main_tree['right_child'] = chk_last_node(left_child)
				return

			main_tree['left_child'] = get_child(left_child)
			splitTree(main_tree['left_child'])
				
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
			if test_data[tree_input['attribute']] < tree_input['chk_value']:
				if isinstance(tree_input['left_child'], dict):
					return get_class(tree_input['left_child'],test_data)
				else:
					return tree_input['left_child']
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
	print("Mean Accuracy using 10 fold validation (Decision Tree with GINI Index): " + str(accuracy))


	# Writing results to a file
	f = open(myname+"result.txt", "w")
	f.write("accuracy: %.4f" % accuracy)
	f.close()


if __name__ == "__main__":
    run_decision_tree()