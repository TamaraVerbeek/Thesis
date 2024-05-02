# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:02:30 2024

@author: tamar
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dual = pd.read_csv('output/infrequent/final/dualprompt/output_dualprompt_ImbalancedTasks.csv_3.csv')
dual1 = pd.read_csv('output/infrequent/final/dualprompt/output_dualprompt_ImbalancedTasks.csv_1.csv')
dual2 = pd.read_csv('output/infrequent/final/dualprompt/output_dualprompt_ImbalancedTasks.csv_2.csv')
dual3 = pd.read_csv('output/infrequent/final/dualprompt/output_dualprompt_ImbalancedTasks.csv_0.csv')
dual4 = pd.read_csv('output/infrequent/final/dualprompt/output_dualprompt_ImbalancedTasks.csv_4.csv')

#dual2 = pd.read_csv('output/infrequent/prompt tuning/output_dualprompt_infreq_prompt_.csv')
dyna = pd.read_csv('output/infrequent/final/dynatraincdd/output_dynatraincdd_imbalance_0.csv')
dyna1 = pd.read_csv('output/infrequent/final/dynatraincdd/output_dynatraincdd_imbalance_1.csv')
dyna2 = pd.read_csv('output/infrequent/final/dynatraincdd/output_dynatraincdd_imbalance_2.csv')
dyna3 = pd.read_csv('output/infrequent/final/dynatraincdd/output_dynatraincdd_imbalance_3.csv')
dyna4 = pd.read_csv('output/infrequent/final/dynatraincdd/output_dynatraincdd_imbalance_4.csv')

gan = pd.read_csv('output/infrequent/final/gan/output_GAN_imbalancedtasks_0.csv')
gan_1 = pd.read_csv('output/infrequent/final/gan/output_GAN_imbalancedtasks_1.csv')
gan_2 = pd.read_csv('output/infrequent/final/gan/output_GAN_imbalancedtasks_2.csv')
gan_3 = pd.read_csv('output/infrequent/final/gan/output_GAN_imbalancedtasks_3.csv')
gan_4 = pd.read_csv('output/infrequent/final/gan/output_GAN_imbalancedtasks_4.csv')

static4 = pd.read_csv('output/infrequent/final/landmark/output_static_infrequent_4.csv')
static3 = pd.read_csv('output/infrequent/final/landmark/output_static_infrequent_3.csv')
static2 = pd.read_csv('output/infrequent/final/landmark/output_static_infrequent_2.csv')
static1 = pd.read_csv('output/infrequent/final/landmark/output_static_infrequent_1.csv')
static0 = pd.read_csv('output/infrequent/final/landmark/output_static_infrequent_0.csv')

lastdrift = pd.read_csv('output/infrequent/final/lastdrift/output_staticlastdrift_infrequent_0.csv')
lastdrift1 = pd.read_csv('output/infrequent/final/lastdrift/output_staticlastdrift_infrequent_1.csv')
lastdrift2 = pd.read_csv('output/infrequent/final/lastdrift/output_staticlastdrift_infrequent_2.csv')
lastdrift3 = pd.read_csv('output/infrequent/final/lastdrift/output_staticlastdrift_infrequent_3.csv')
lastdrift4 = pd.read_csv('output/infrequent/final/lastdrift/output_staticlastdrift_infrequent_4.csv')

tfcl0 = pd.read_csv('output/infrequent/final/tfcl/prediction_results.csv')
tfcl1 = pd.read_csv('output/infrequent/final/tfcl/prediction_results1.csv')
tfcl2 = pd.read_csv('output/infrequent/final/tfcl/prediction_results2.csv')
tfcl3 = pd.read_csv('output/infrequent/final/tfcl/prediction_results3.csv')
tfcl4 = pd.read_csv('output/infrequent/final/tfcl/prediction_results4.csv')

def get0and1(dataframe):
    results = []
    for row in dataframe.iterrows():
        if row[1][1] == row[1][2]:
            results.append(1)
        else:
            results.append(0)
    return results

drift0 = get0and1(lastdrift)
drift1 = get0and1(lastdrift1)
drift2 = get0and1(lastdrift2)
drift3 = get0and1(lastdrift3)
drift4 = get0and1(lastdrift4)

dyna_0 = get0and1(dyna)
dyna_1 = get0and1(dyna1)
dyna_2 = get0and1(dyna2)
dyna_3 = get0and1(dyna3)
dyna_4 = get0and1(dyna4)

land0 = get0and1(static0)
land1 = get0and1(static1)
land2 = get0and1(static2)
land3 = get0and1(static3)
land4 = get0and1(static4)

def get0and1_other(dataframe):
    results = []
    for row in dataframe.iterrows():
        if row[1][0] == row[1][1]:
            results.append(1)
        else:
            results.append(0)
    return results
        
tfcl_0 = get0and1_other(tfcl0)
tfcl_1 = get0and1_other(tfcl1)
tfcl_2 = get0and1_other(tfcl2)
tfcl_3 = get0and1_other(tfcl3)
tfcl_4 = get0and1_other(tfcl4)

def calculate_percentage_of_ones(data):
    window_size = 350
    result = []

    for i in range(len(data)):
        start_index = max(0, i - window_size + 1)
        window = data[start_index:i+1]
        percentage_of_ones = sum(window) / len(window) * 100
        result.append(percentage_of_ones)

    return result

x = calculate_percentage_of_ones(dyna_0)
y = range(500, len(x)+500)
r1 = calculate_percentage_of_ones(dyna_1)
r2 = calculate_percentage_of_ones(dyna_2)
r3 = calculate_percentage_of_ones(dyna_3)
r4 = calculate_percentage_of_ones(dyna_4)
acc = [sum(x)/len(x) for x in [x, r1,r2, r3, r4]] 
avg_acc_lastdrift = np.mean(acc)
print(f'Accuracies of DynaTrainCDD: {acc}')
print(f'Average accuracy of DynaTrainCDD: {avg_acc_lastdrift}')

x2 = calculate_percentage_of_ones(land0)
y2 = range(500, len(x2)+500)
r1 = calculate_percentage_of_ones(land1)
r2 = calculate_percentage_of_ones(land2)
r3 = calculate_percentage_of_ones(land3)
r4 = calculate_percentage_of_ones(land4)
acc = [sum(x)/len(x) for x in [x2, r1,r2, r3, r4]] 
avg_acc_lastdrift = np.mean(acc)
print(f'Accuracies of landmark: {acc}')
print(f'Average accuracy of landmark: {avg_acc_lastdrift}')

x3 = calculate_percentage_of_ones(drift0)
y3 = range(500, len(x3)+500)
r1 = calculate_percentage_of_ones(drift1)
r2 = calculate_percentage_of_ones(drift2)
r3 = calculate_percentage_of_ones(drift3)
r4 = calculate_percentage_of_ones(drift4)
acc = [sum(x)/len(x) for x in [x, r1,r2, r3, r4]] 
avg_acc_lastdrift = np.mean(acc)
print(f'Accuracies of last drifts: {acc}')
print(f'Average accuracy of last drifts: {avg_acc_lastdrift}')

x4= dual.iloc[:,0]
y4 = dual.iloc[:,1]
dual_1 = dual1.iloc[:,1]
dual_2 = dual2.iloc[:,1]
dual_3 = dual3.iloc[:,1]
dual_4 = dual4.iloc[:,1]
acc = [sum(x)/len(x) for x in [y4, dual_1, dual_2, dual_3, dual_4]] 
avg_acc_landmark = np.mean(acc)
print(f'Accuracies of dualprompt: {acc}')
print(f'Average accuracy of dualprompt: {avg_acc_landmark}')

x5 = calculate_percentage_of_ones(gan['0'])
y5 = range(500, len(x5)+500)
r1 = calculate_percentage_of_ones(gan_1['0'])
r2 = calculate_percentage_of_ones(gan_2['0'])
r3 = calculate_percentage_of_ones(gan_3['0'])
r4 = calculate_percentage_of_ones(gan_4['0'])
acc = [sum(x)/len(x) for x in [x5, r1,r2, r3, r4]] 
avg_acc_gan = np.mean(acc)
print(f'Accuracies of GAN: {acc}')
print(f'Average accuracy of GAN: {avg_acc_gan}')

#x6 = dual2.iloc[:,0]
#y6 = dual2.iloc[:,1]
x7 = calculate_percentage_of_ones(tfcl_0)
y7 = range(500, len(x7)+500)
r1 = calculate_percentage_of_ones(tfcl_1)
r2 = calculate_percentage_of_ones(tfcl_2)
r3 = calculate_percentage_of_ones(tfcl_3)
r4 = calculate_percentage_of_ones(tfcl_4)
acc = [sum(x)/len(x) for x in [x7, r1,r2, r3, r4]] 
avg_acc_lastdrift = np.mean(acc)
print(f'Accuracies of tfcl: {acc}')
print(f'Average accuracy of tfcl: {avg_acc_lastdrift}')

plt.figure(figsize=(20,5))
plt.plot(y, x, linestyle='-', linewidth=1, color = 'red', label = 'DynaTrainCDD')  
plt.plot(y2, x2, linestyle='-', linewidth=1, color = 'yellow', label = 'Landmark')  
plt.plot(y3, x3, linestyle='-', linewidth=1, color = 'orange', label = 'Last drift') 
#plt.plot(x6, y6*100, linestyle='-', linewidth=1, color = 'green', label = 'Prompt tuning (Pro-T)') 
plt.plot(x4, y4*100, linestyle='-', linewidth=1, color = 'blue', label = 'Dualprompt')
plt.plot(y5, x5, linestyle='-', linewidth=1, color = 'pink', label = 'GAN')
plt.plot(y7, x7, linestyle='-', linewidth=1, color = 'green', label = 'tfcl') 


concept_drift = [2724, 5469, 8501, 11773, 14769, 17474, 19976, 22804, 25528, 28030]
for index in concept_drift:
    plt.axvline(x=index, color='red', linestyle='--', linewidth=1)  # Adjust color, linestyle, and linewidth as needed

c = [0, 2724, 5469, 8501, 11773, 14769, 17474, 19976, 22804, 25528, 28030, len(x)+500]
plt.axvspan(c[0], c[1], alpha=0.3, color = 'blue', label = 'Task 1')
plt.axvspan(c[5], c[6], alpha=0.3, color = 'blue')
plt.axvspan(c[8], c[9], alpha=0.3, color = 'blue')
plt.axvspan(c[10], c[11], alpha=0.3, color = 'blue')

plt.axvspan(c[1], c[2], alpha=0.3, color = 'green', label = 'Task 2')
plt.axvspan(c[7], c[8], alpha=0.3, color = 'green')
plt.axvspan(c[9], c[10], alpha=0.3, color = 'green')

plt.axvspan(c[2], c[3], alpha=0.3, color = 'purple', label = 'Task 3')
plt.axvspan(c[4], c[5], alpha=0.3, color = 'purple')
plt.axvspan(c[6], c[7], alpha=0.3, color = 'purple')

plt.axvspan(c[3], c[4], alpha=0.3, color = 'yellow', label = 'Task 4')


plt.xticks(range(0, len(x), 5000))
plt.xlabel('Event')
plt.ylabel('Accuracy')
plt.title('ImbalancedTasks dataset')
plt.grid(True)  # Add grid lines (optional)
plt.legend()
plt.show()

avg_acc = sum(x)/len(x) #dynatraincdd
print(f'DynaTrainCDD: {avg_acc}')
avg_acc = sum(x2)/len(x2) #landmark
print(f'Landmark: {avg_acc}')
avg_acc = sum(x3)/len(x3) #last drift
print(f'Last Drift: {avg_acc}')
avg_acc = sum(y4)/len(y4) #dualprompt
print(f'Dualprompt: {avg_acc*100}')
avg_acc = sum(x5)/len(x5) #gan
print(f'GAN: {avg_acc}')
#avg_acc = sum(y6)/len(y6) #dualprompt
#print(avg_acc*100)
avg_acc = sum(x7)/len(x7) #tfcl
print(f'tfcl: {avg_acc}')
