import numpy as np

# calculating accuracy 
def get_acc(gt,pred):
    assert len(gt)== len(pred)
    correct = 0
    for i in range(len(gt)):
        if gt[i]==pred[i]:
            correct+=1
            
    return (1.0*correct)/len(gt)

gt_all = [line.strip().split(',')[3] for line in open('cb513test_solution.csv').readlines()]
predictions = [line.strip().split(',')[1] for line in open('hilbert_solution.csv').readlines()[1:]]
acc_list = []

for gt,pred in zip(gt_all,predictions):
	if len(gt) == len(pred):
		acc = get_acc(gt,pred)
		acc_list.append(acc)
		print(gt)
		print()
		print(pred)
		print('\n')
print ('mean accuracy is', np.mean(acc_list))