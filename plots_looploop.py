# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:24:30 2024

@author: tamar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

lastdrift0 = pd.read_csv('output/bigloop_dataset/final/lastdrift/output_lastdrift_loop_0.csv')
lastdrift1 = pd.read_csv('output/bigloop_dataset/final/lastdrift/output_lastdrift_loop_1.csv')
lastdrift2 = pd.read_csv('output/bigloop_dataset/final/lastdrift/output_lastdrift_loop_2.csv')
lastdrift3 = pd.read_csv('output/bigloop_dataset/final/lastdrift/output_lastdrift_loop_3.csv')
lastdrift4 = pd.read_csv('output/bigloop_dataset/final/lastdrift/output_lastdrift_loop_4.csv')

static0 = pd.read_csv('output/bigloop_dataset/final/landmark/output_static_looploop_0.csv') 
static1 = pd.read_csv('output/bigloop_dataset/final/landmark/output_static_looploop_1.csv') 
static2 = pd.read_csv('output/bigloop_dataset/final/landmark/output_static_looploop_2.csv') 
static3 = pd.read_csv('output/bigloop_dataset/final/landmark/output_static_looploop_3.csv') 
static4 = pd.read_csv('output/bigloop_dataset/final/landmark/output_static_looploop_4.csv') 

dual = pd.read_csv('output/bigloop_dataset/final/dualprompt/output250_RecurrentTasks.csv_0.csv')
dual1 = pd.read_csv('output/bigloop_dataset/final/dualprompt/output_dualprompt_RecurrentTasks.csv_1.csv')
dual2 = pd.read_csv('output/bigloop_dataset/final/dualprompt/output_dualprompt_RecurrentTasks.csv_2.csv')
dual3 = pd.read_csv('output/bigloop_dataset/final/dualprompt/output_dualprompt_RecurrentTasks.csv_3.csv')
dual4 = pd.read_csv('output/bigloop_dataset/final/dualprompt/output_dualprompt_RecurrentTasks.csv_4.csv')

gan = pd.read_csv('output/bigloop_dataset/final/gan/output_GAN_recurrenttasks_0.csv')
gan_1 = pd.read_csv('output/bigloop_dataset/final/gan/output_GAN_recurrenttasks_1.csv')
gan_2 = pd.read_csv('output/bigloop_dataset/final/gan/output_GAN_recurrenttasks_2.csv')
gan_3 = pd.read_csv('output/bigloop_dataset/final/gan/output_GAN_recurrenttasks_3.csv')
gan_4 = pd.read_csv('output/bigloop_dataset/final/gan/output_GAN_recurrenttasks_4.csv')

dyna = pd.read_csv('output/bigloop_dataset/final/dynatraincdd/output_dynatraincdd_looploop_0.csv')
dyna1 = pd.read_csv('output/bigloop_dataset/final/dynatraincdd/output_dynatraincdd_looploop_1.csv')
dyna2 = pd.read_csv('output/bigloop_dataset/final/dynatraincdd/output_dynatraincdd_looploop_2.csv')
dyna3 = pd.read_csv('output/bigloop_dataset/final/dynatraincdd/output_dynatraincdd_looploop_3.csv')
dyna4 = pd.read_csv('output/bigloop_dataset/final/dynatraincdd/output_dynatraincdd_looploop_4.csv')


tfcl = pd.read_csv('output/bigloop_dataset/final/tfcl/prediction_results.csv')
tfcl1 = pd.read_csv('output/bigloop_dataset/final/tfcl/prediction_results1.csv')
tfcl2 = pd.read_csv('output/bigloop_dataset/final/tfcl/prediction_results2.csv')
tfcl3 = pd.read_csv('output/bigloop_dataset/final/tfcl/prediction_results3.csv')
tfcl4 = pd.read_csv('output/bigloop_dataset/final/tfcl/prediction_results4.csv')


def get0and1(dataframe):
    results = []
    for row in dataframe.iterrows():
        if row[1][1] == row[1][2]:
            results.append(1)
        else:
            results.append(0)
    return results

drift0 = get0and1(lastdrift0)
drift1 = get0and1(lastdrift1)
drift2 = get0and1(lastdrift2)
drift3 = get0and1(lastdrift3)
drift4 = get0and1(lastdrift4)

static_0 = get0and1(static0)
static_1 = get0and1(static1)
static_2 = get0and1(static2)
static_3 = get0and1(static3)
static_4 = get0and1(static4)

dyna_0 = get0and1(dyna)
dyna_1 = get0and1(dyna1)
dyna_2 = get0and1(dyna2)
dyna_3 = get0and1(dyna3)
dyna_4 = get0and1(dyna4)        


def get0and1_other(dataframe):
    results = []
    for row in dataframe.iterrows():
        if row[1][0] == row[1][1]:
            results.append(1)
        else:
            results.append(0)
    return results

tfcl_0 = get0and1_other(tfcl)
tfcl_1 = get0and1_other(tfcl1)
tfcl_2 = get0and1_other(tfcl2)
tfcl_3 = get0and1_other(tfcl3)
tfcl_4 = get0and1_other(tfcl4)

def calculate_percentage_of_ones(data):
    window_size = 250
    result = []

    for i in range(len(data)):
        start_index = max(0, i - window_size + 1)
        window = data[start_index:i+1]
        percentage_of_ones = sum(window) / len(window) * 100
        result.append(percentage_of_ones)

    return result

x = calculate_percentage_of_ones(drift0)
y = range(500, len(x)+500)
r1 = calculate_percentage_of_ones(drift1)
r2 = calculate_percentage_of_ones(drift2)
r3 = calculate_percentage_of_ones(drift3)
r4 = calculate_percentage_of_ones(drift4)
acc = [sum(x)/len(x) for x in [x, r1,r2, r3, r4]] 
avg_acc_lastdrift = np.mean(acc)
print(f'Accuracies of last drifts: {acc}')
print(f'Average accuracy of last drifts: {avg_acc_lastdrift}')

x2 = calculate_percentage_of_ones(static_0)
y2 = range(500, len(x2)+500)
r1 = calculate_percentage_of_ones(static_1)
r2 = calculate_percentage_of_ones(static_2)
r3 = calculate_percentage_of_ones(static_3)
r4 = calculate_percentage_of_ones(static_4)
acc = [sum(x)/len(x) for x in [x2, r1, r2, r3, r4]] 
avg_acc_landmark = np.mean(acc)
print(f'Accuracies of landmark: {acc}')
print(f'Average accuracy of landmark: {avg_acc_landmark}')

x3 = dual1.iloc[:,0]
y3 = dual1.iloc[:,1]
dual_1 = dual.iloc[:,1]
dual_2 = dual2.iloc[:,1]
dual_3 = dual3.iloc[:,1]
dual_4 = dual4.iloc[:,1]
acc = [sum(x)/len(x) for x in [y3, dual_1, dual_2, dual_3, dual_4]] 
avg_acc_landmark = np.mean(acc)
print(f'Accuracies of dualprompt: {acc}')
print(f'Average accuracy of dualprompt: {avg_acc_landmark}')

x4 = calculate_percentage_of_ones(gan['0'])
y4 = range(500, len(x4)+500)
r1 = calculate_percentage_of_ones(gan_1['0'])
r2 = calculate_percentage_of_ones(gan_2['0'])
r3 = calculate_percentage_of_ones(gan_3['0'])
r4 = calculate_percentage_of_ones(gan_4['0'])
acc = [sum(x)/len(x) for x in [x4, r1,r2, r3, r4]] 
avg_acc_gan = np.mean(acc)
print(f'Accuracies of GAN: {acc}')
print(f'Average accuracy of GAN: {avg_acc_gan}')


x5 = calculate_percentage_of_ones(dyna_0)
y5 = range(500, len(x5)+500)
r1 = calculate_percentage_of_ones(dyna_1)
r2 = calculate_percentage_of_ones(dyna_2)
r3 = calculate_percentage_of_ones(dyna_3)
r4 = calculate_percentage_of_ones(dyna_4)
acc = [sum(x)/len(x) for x in [x5, r1,r2, r3, r4]] 
avg_acc_lastdrift = np.mean(acc)
print(f'Accuracies of DynaTrainCDD: {acc}')
print(f'Average accuracy of DynaTrainCDD: {avg_acc_lastdrift}')

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


concept_drift =  [2725, 5757, 9075, 12071, 15502, 18207, 21268, 24633, 27758, 31178, 33822, 36838, 40221, 42722, 45223, 47725, 50563, 53523, 56323, 59190, 61790, 64512, 67380, 70182, 73124, 75720, 78222, 80724, 83226, 85728, 88230, 90732, 93234, 95736, 98238] #combi
plt.figure(figsize=(20,5))
plt.plot(y, x, linestyle='-', linewidth=1, color = 'orange', label = 'Last drift')  
plt.plot(y2, x2, linestyle='-', linewidth=1, color = 'yellow', label = 'Landmark')  
plt.plot(y4, x4, linestyle='-', linewidth=1, color = 'pink', label = 'GAN')
plt.plot(y5, x5, linestyle='-', linewidth=1, color = 'red', label = 'DynaTrainCDD')
plt.plot(y7, x7, linestyle='-', linewidth=1, color = 'green', label = 'tfcl') 
plt.plot(x3, y3*100, linestyle='-', linewidth=1, color = 'blue', label = 'Dualprompt')  

for index in concept_drift:
    plt.axvline(x=index, color='red', linestyle='--', linewidth=1)  # Adjust color, linestyle, and linewidth as needed
c = [500, 2725, 5757, 9075, 12071, 15502, 18207, 21268, 24633, 27758, 31178, 33822, 36838, 40221, 42722, 45223, 47725, 50563, 53523, 56323, 59190, 61790, 64512, 67380, 70182, 73124, 75720, 78222, 80724, 83226, 85728, 88230, 90732, 93234, 95736, 98238, len(x)+500] #combi

plt.axvspan(c[0], c[1], alpha=0.3, color = 'blue', label = 'Task 1')
plt.axvspan(c[5], c[6], alpha=0.3, color = 'blue')
plt.axvspan(c[10], c[11], alpha=0.3, color = 'blue')
plt.axvspan(c[15], c[16], alpha=0.3, color = 'blue')
plt.axvspan(c[20], c[21], alpha=0.3, color = 'blue')
plt.axvspan(c[25], c[26], alpha=0.3, color = 'blue')
plt.axvspan(c[30], c[31], alpha=0.3, color = 'blue')
plt.axvspan(c[35], c[36], alpha=0.3, color = 'blue')

plt.axvspan(c[1], c[2], alpha=0.3, color = 'green', label = 'Task 2')
plt.axvspan(c[3], c[4], alpha=0.3, color = 'green')
plt.axvspan(c[6], c[7], alpha=0.3, color = 'green')
plt.axvspan(c[8], c[9], alpha=0.3, color = 'green')
plt.axvspan(c[11], c[12], alpha=0.3, color = 'green')
plt.axvspan(c[13], c[14], alpha=0.3, color = 'green')
plt.axvspan(c[16], c[17], alpha=0.3, color = 'green')
plt.axvspan(c[18], c[19], alpha=0.3, color = 'green')
plt.axvspan(c[21], c[22], alpha=0.3, color = 'green')
plt.axvspan(c[23], c[24], alpha=0.3, color = 'green')
plt.axvspan(c[26], c[27], alpha=0.3, color = 'green')
plt.axvspan(c[28], c[29], alpha=0.3, color = 'green')
plt.axvspan(c[31], c[32], alpha=0.3, color = 'green')
plt.axvspan(c[33], c[34], alpha=0.3, color = 'green')

plt.axvspan(c[2], c[3], alpha=0.3, color = 'purple', label = 'Task 3')
plt.axvspan(c[4], c[5], alpha=0.3, color = 'purple')
plt.axvspan(c[7], c[8], alpha=0.3, color = 'purple')
plt.axvspan(c[9], c[10], alpha=0.3, color = 'purple')
plt.axvspan(c[12], c[13], alpha=0.3, color = 'purple')
plt.axvspan(c[14], c[15], alpha=0.3, color = 'purple')
plt.axvspan(c[17], c[18], alpha=0.3, color = 'purple')
plt.axvspan(c[19], c[20], alpha=0.3, color = 'purple')
plt.axvspan(c[22], c[23], alpha=0.3, color = 'purple')
plt.axvspan(c[24], c[25], alpha=0.3, color = 'purple')
plt.axvspan(c[27], c[28], alpha=0.3, color = 'purple')
plt.axvspan(c[29], c[30], alpha=0.3, color = 'purple')
plt.axvspan(c[32], c[33], alpha=0.3, color = 'purple')
plt.axvspan(c[34], c[35], alpha=0.3, color = 'purple')

plt.xticks(range(0, len(x5), 5000))
plt.xlabel('Event')
plt.ylabel('Accuracy')
plt.title('RecurrentTasks dataset')
plt.grid(True)  # Add grid lines (optional)
plt.legend()
plt.show()

avg_acc = sum(x)/len(x)
print(f'Last drift: {avg_acc}')
avg_acc = sum(x2)/len(x2)
print(f'Landmark: {avg_acc}')

avg_acc = sum(x4)/len(x4)
print(f'GAN: {avg_acc}')
avg_acc = sum(x5)/len(x5)
print(f'DynaTrainCDD: {avg_acc}')

avg_acc = sum(x7)/len(x7) 
print(f'tfcl: {avg_acc}')
avg_acc = sum(y3)/len(y3)
print(f'Dualprompt: {avg_acc*100}')
