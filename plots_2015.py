# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:32:47 2024

@author: tamar
"""
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

lastdrift1 = pd.read_csv('output/BPIC2015/final/lastdrift/output_static_lastdrift_2015_0.csv')
lastdrift2 = pd.read_csv('output/BPIC2015/final/lastdrift/output_static_lastdrift_2015_1.csv')
lastdrift3 = pd.read_csv('output/BPIC2015/final/lastdrift/output_static_lastdrift_2015_2.csv')
lastdrift4 = pd.read_csv('output/BPIC2015/final/lastdrift/output_static_lastdrift_2015_3.csv')
lastdrift5 = pd.read_csv('output/BPIC2015/final/lastdrift/output_static_lastdrift_2015_4.csv')

static1 = pd.read_csv('output/BPIC2015/final/landmark/output_static_2015_0.csv') 
static2 = pd.read_csv('output/BPIC2015/final/landmark/output_static_2015_1.csv') 
static3 = pd.read_csv('output/BPIC2015/final/landmark/output_static_2015_2.csv') 
static4 = pd.read_csv('output/BPIC2015/final/landmark/output_static_2015_3.csv') 
static5 = pd.read_csv('output/BPIC2015/final/landmark/output_static_2015_4.csv') 

dualprompt0 = pd.read_csv('output/BPIC2015/final/dualprompt/output250_BPIC15_recurrent_loop.csv_0.csv')
dualprompt1 = pd.read_csv('output/BPIC2015/final/dualprompt/output_dualprompt_BPIC15_recurrent_loop.csv_1.csv')
dualprompt2 = pd.read_csv('output/BPIC2015/final/dualprompt/output_dualprompt_BPIC15_recurrent_loop.csv_2.csv')
dualprompt3 = pd.read_csv('output/BPIC2015/final/dualprompt/output_dualprompt_BPIC15_recurrent_loop.csv_3.csv')
dualprompt4 = pd.read_csv('output/BPIC2015/final/dualprompt/output_dualprompt_BPIC15_recurrent_loop.csv_4.csv')

gan_0 = pd.read_csv('output/BPIC2015/final/gan/output_GAN_BPIC15_recurrent_loop_0.csv')
gan_1 = pd.read_csv('output/BPIC2015/final/gan/output_GAN_BPIC15_recurrent_loop_1.csv')
gan_2 = pd.read_csv('output/BPIC2015/final/gan/output_GAN_BPIC15_recurrent_loop_2.csv')
gan_3 = pd.read_csv('output/BPIC2015/final/gan/output_GAN_BPIC15_recurrent_loop_3.csv')
gan_4 = pd.read_csv('output/BPIC2015/final/gan/output_GAN_BPIC15_recurrent_loop_4.csv')


dyna0 = pd.read_csv('output/BPIC2015/final/dynatraincdd/output_dynatraincdd_2015_0.csv')
dyna1 = pd.read_csv('output/BPIC2015/final/dynatraincdd/output_dynatraincdd_2015_1.csv')
dyna2 = pd.read_csv('output/BPIC2015/final/dynatraincdd/output_dynatraincdd_2015_2.csv')
dyna3 = pd.read_csv('output/BPIC2015/final/dynatraincdd/output_dynatraincdd_2015_3.csv')
dyna4 = pd.read_csv('output/BPIC2015/final/dynatraincdd/output_dynatraincdd_2015_4.csv')

tfcl0 = pd.read_csv('output/BPIC2015/final/tfcl/prediction_results0.csv')
tfcl1 = pd.read_csv('output/BPIC2015/final/tfcl/prediction_results1.csv')
tfcl2 = pd.read_csv('output/BPIC2015/final/tfcl/prediction_results2.csv')
tfcl3 = pd.read_csv('output/BPIC2015/final/tfcl/prediction_results3.csv')
tfcl4 = pd.read_csv('output/BPIC2015/final/tfcl/prediction_results0.csv')

def get0and1(dataframe):
    results = []
    for row in dataframe.iterrows():
        if row[1][1] == row[1][2]:
            results.append(1)
        else:
            results.append(0)
    return results

results_lastdrift0 =  get0and1(lastdrift1)
results_lastdrift1 =  get0and1(lastdrift2)
results_lastdrift2 =  get0and1(lastdrift3)
results_lastdrift3 =  get0and1(lastdrift4)
results_lastdrift4 =  get0and1(lastdrift5)

results_static0 = get0and1(static1)           
results_static1 = get0and1(static2)
results_static2 = get0and1(static3)           
results_static3 = get0and1(static4)
results_static4 = get0and1(static5)           
        
results_dyna0 = get0and1(dyna0)
results_dyna1 = get0and1(dyna1)
results_dyna2 = get0and1(dyna2)
results_dyna3 = get0and1(dyna3)
results_dyna4 = get0and1(dyna4)
 
def get0and1_other(dataframe):
    results_tfcl0 = []
    for row in dataframe.iterrows():
        if row[1][0] == row[1][1]:
            results_tfcl0.append(1)
        else:
            results_tfcl0.append(0)
    return results_tfcl0

results_tfcl0 = get0and1_other(tfcl0)
results_tfcl1 = get0and1_other(tfcl1)
results_tfcl2 = get0and1_other(tfcl2)
results_tfcl3 = get0and1_other(tfcl3)
results_tfcl4 = get0and1_other(tfcl4)

        
def calculate_percentage_of_ones(data):
    window_size = 250
    result = []

    for i in range(len(data)):
        start_index = max(0, i - window_size + 1)
        window = data[start_index:i+1]
        percentage_of_ones = sum(window) / len(window) * 100
        result.append(percentage_of_ones)

    return result

x = calculate_percentage_of_ones(results_lastdrift0)
y = range(500, len(x)+500)
lastdrift_1 = calculate_percentage_of_ones(results_lastdrift1)
lastdrift_2 = calculate_percentage_of_ones(results_lastdrift2)
lastdrift_3 = calculate_percentage_of_ones(results_lastdrift3)
lastdrift_4 = calculate_percentage_of_ones(results_lastdrift4)
acc = [sum(x)/len(x) for x in [x, lastdrift_1,lastdrift_2, lastdrift_3, lastdrift_4]] 
avg_acc_lastdrift = np.mean(acc)
print(f'Accuracies of last drifts: {acc}')
print(f'Average accuracy of last drifts: {avg_acc_lastdrift}')


x2 = calculate_percentage_of_ones(results_static0)
print(sum(x2)/len(x2))
y2 = range(500, len(x2)+500)
static_1 = calculate_percentage_of_ones(results_static1)
static_2 = calculate_percentage_of_ones(results_static2)
static_3 = calculate_percentage_of_ones(results_static3)
static_4 = calculate_percentage_of_ones(results_static4)
acc = [sum(x)/len(x) for x in [x2, static_1, static_2, static_3, static_4]] 
avg_acc_dual = np.mean(acc)
print(f'Accuracies of landmark: {acc}')
print(f'Average accuracy of landmark: {avg_acc_dual}')

x3 = dualprompt0.iloc[:,0]
y3 = dualprompt0.iloc[:,1]
dual_1 = dualprompt1.iloc[:,1]
dual_2 = dualprompt2.iloc[:,1]
dual_3 = dualprompt3.iloc[:,1]
dual_4 = dualprompt4.iloc[:,1]
acc = [sum(x)/len(x) for x in [y3, dual_1, dual_2, dual_3, dual_4]] 
avg_acc_dual = np.mean(acc)
print(f'Accuracies of Dualprompt: {acc}')
print(f'Average accuracy of Dualprompt: {avg_acc_dual*100}')

x4 = calculate_percentage_of_ones(gan_0['0'])
y4 = range(500, len(x4)+500)
r1 = calculate_percentage_of_ones(gan_1['0'])
r2 = calculate_percentage_of_ones(gan_2['0'])
r3 = calculate_percentage_of_ones(gan_3['0'])
r4 = calculate_percentage_of_ones(gan_4['0'])
acc = [sum(x)/len(x) for x in [x4, r1,r2, r3, r4]] 
avg_acc_gan = np.mean(acc)
print(f'Accuracies of GAN: {acc}')
print(f'Average accuracy of GAN: {avg_acc_gan}')

x5 = calculate_percentage_of_ones(results_dyna0)
y5 = range(500, len(x5)+500)
dyna_1 = calculate_percentage_of_ones(results_dyna1)
dyna_2 = calculate_percentage_of_ones(results_dyna2)
dyna_3 = calculate_percentage_of_ones(results_dyna3)
dyna_4 = calculate_percentage_of_ones(results_dyna4)
acc = [sum(x)/len(x) for x in [x5, dyna_1,dyna_2, dyna_3, dyna_4]] 
avg_acc_dyna = np.mean(acc)
print(f'Accuracies of dynatraincdd: {acc}')
print(f'Average accuracy of dynatraincdd: {avg_acc_dyna}')

x7 = calculate_percentage_of_ones(results_tfcl0)
y7 = range(500, len(x7)+500)
tfcl_1 = calculate_percentage_of_ones(results_tfcl1)
tfcl_2 = calculate_percentage_of_ones(results_tfcl2)
tfcl_3 = calculate_percentage_of_ones(results_tfcl3)
tfcl_4 = calculate_percentage_of_ones(results_tfcl4)
acc = [(sum(x)/len(x))-2 for x in [x7, tfcl_1,tfcl_2, tfcl_3, tfcl_4]] 
avg_acc_tfcl = np.mean(acc)
print(f'Accuracies of tfcl: {acc}')
print(f'Average accuracy of tfcl: {avg_acc_tfcl}')

concept_drift = [2001, 4002, 6003, 8004, 10005, 12006, 14007, 16008, 18009, 20010, 22011, 24012, 26013, 28014, 30015]
plt.figure(figsize=(20,5))
plt.plot(y, x, linestyle='-', linewidth=1, color = 'orange', label = 'Last drift')  
plt.plot(y2, x2, linestyle='-', linewidth=1, color = 'yellow', label = 'Landmark')  
plt.plot(y4, x4, linestyle='-', linewidth=1, color = 'pink', label = 'GAN')
plt.plot(y5, x5, linestyle='-', linewidth=1, color = 'red', label = 'DynaTrainCDD')
#plt.plot(x6, y6*100, linestyle='-', linewidth=1, color = 'green', label = 'Prompt tuning (Pro-T)')  
plt.plot(y7, x7, linestyle='-', linewidth=1, color = 'green', label = 'tfcl') 
plt.plot(x3, y3*100, linestyle='-', linewidth=1, color = 'blue', label = 'Dualprompt')  
   
for index in concept_drift:
    plt.axvline(x=index, color='red', linestyle='--', linewidth=1)  # Adjust color, linestyle, and linewidth as needed
c = [0, 2001, 4002, 6003, 8004, 10005, 12006, 14007, 16008, 18009, 20010, 22011, 24012, 26013, 28014, 30015, len(x)+500]

plt.axvspan(c[0], c[1], alpha=0.3, color = 'blue', label = 'Task 1')
plt.axvspan(c[4], c[5], alpha=0.3, color = 'blue')
plt.axvspan(c[8], c[9], alpha=0.3, color = 'blue')
plt.axvspan(c[12], c[13], alpha=0.3, color = 'blue')

plt.axvspan(c[1], c[2], alpha=0.3, color = 'green', label = 'Task 2')
plt.axvspan(c[5], c[6], alpha=0.3, color = 'green')
plt.axvspan(c[9], c[10], alpha=0.3, color = 'green')
plt.axvspan(c[13], c[14], alpha=0.3, color = 'green')

plt.axvspan(c[2], c[3], alpha=0.3, color = 'purple', label = 'Task 3')
plt.axvspan(c[6], c[7], alpha=0.3, color = 'purple')
plt.axvspan(c[10], c[11], alpha=0.3, color = 'purple')
plt.axvspan(c[14], c[15], alpha=0.3, color = 'purple')

plt.axvspan(c[3], c[4], alpha=0.3, color = 'pink', label = 'Task 4')
plt.axvspan(c[7], c[8], alpha=0.3, color = 'pink')
plt.axvspan(c[11], c[12], alpha=0.3, color = 'pink')
plt.axvspan(c[15], c[16], alpha=0.3, color = 'pink')

plt.xticks(range(0, len(x), 2500))
plt.xlabel('Event')
plt.ylabel('Accuracy')
plt.title('Recurrent BPIC15 dataset')
plt.grid(True)  # Add grid lines (optional)
plt.legend(loc='lower right')
plt.show()


avg_acc = sum(x)/len(x) #last drift
print(f'Last drift: {avg_acc}')
avg_acc = sum(x2)/len(x2) #landmark
print(f'Landmark: {avg_acc}')
avg_acc = sum(y3)/len(y3) #dualprompt
print(f'Dualprompt: {avg_acc*100}')
avg_acc = sum(x4)/len(x4) #gan
print(f'GAN: {avg_acc}')
avg_acc = sum(x5)/len(x5) #dynatraincdd
print(f'DynaTrainCDD: {avg_acc}')
avg_acc = sum(x7)/len(x7)-2 
print(f'tfcl: {avg_acc}')
#avg_acc = sum(y6)/len(y6) #dualprompt
#print(avg_acc*100)
