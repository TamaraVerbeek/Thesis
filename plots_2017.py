# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:20:01 2024

@author: tamar
"""

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

lastdrift0 = pd.read_csv('output/BPIC2017/final/lastdrift/output_staticlastdrift_2017_0.csv')
lastdrift1 = pd.read_csv('output/BPIC2017/final/lastdrift/output_staticlastdrift_2017_1.csv')
lastdrift2 = pd.read_csv('output/BPIC2017/final/lastdrift/output_staticlastdrift_2017_2.csv')
lastdrift3 = pd.read_csv('output/BPIC2017/final/lastdrift/output_staticlastdrift_2017_3.csv')
lastdrift4 = pd.read_csv('output/BPIC2017/final/lastdrift/output_staticlastdrift_2017_4.csv')

static0 = pd.read_csv('output/BPIC2017/final/landmark/output_static_2017_0.csv') 
static1 = pd.read_csv('output/BPIC2017/final/landmark/output_static_2017_1.csv') 
static2 = pd.read_csv('output/BPIC2017/final/landmark/output_static_2017_2.csv') 
static3 = pd.read_csv('output/BPIC2017/final/landmark/output_static_2017_3.csv') 
static4 = pd.read_csv('output/BPIC2017/final/landmark/output_static_2017_4.csv') 

dual0 = pd.read_csv('output/BPIC2017/final/dualprompt/output250_BPIC17.csv_0.csv')
dual1 = pd.read_csv('output/BPIC2017/final/dualprompt/output_dualprompt_BPIC17.csv_1.csv')
dual2 = pd.read_csv('output/BPIC2017/final/dualprompt/output_dualprompt_BPIC17.csv_2.csv')
dual3 = pd.read_csv('output/BPIC2017/final/dualprompt/output_dualprompt_BPIC17.csv_3.csv')
dual4 = pd.read_csv('output/BPIC2017/final/dualprompt/output_dualprompt_BPIC17.csv_4.csv')

gan = pd.read_csv('output/BPIC2017/final/gan/output_GAN_BPIC17_0.csv')
gan_1 = pd.read_csv('output/BPIC2017/final/gan/output_GAN_BPIC17_1.csv')
gan_2 = pd.read_csv('output/BPIC2017/final/gan/output_GAN_BPIC17_2.csv')
gan_3 = pd.read_csv('output/BPIC2017/final/gan/output_GAN_BPIC17_3.csv')
gan_4 = pd.read_csv('output/BPIC2017/final/gan/output_GAN_BPIC17_4.csv')

dyna = pd.read_csv('output/BPIC2017/final/dynatraincdd/output_dynatraincdd_2017_0.csv')
dyna1 = pd.read_csv('output/BPIC2017/final/dynatraincdd/output_dynatraincdd_2017_1.csv')
dyna2 = pd.read_csv('output/BPIC2017/final/dynatraincdd/output_dynatraincdd_2017_2.csv')
dyna3 = pd.read_csv('output/BPIC2017/final/dynatraincdd/output_dynatraincdd_2017_3.csv')
dyna4 = pd.read_csv('output/BPIC2017/final/dynatraincdd/output_dynatraincdd_2017_4.csv')

tfcl = pd.read_csv('output/BPIC2017/final/tfcl/prediction_results.csv')
tfcl1 = pd.read_csv('output/BPIC2017/final/tfcl/prediction_results1.csv')
tfcl2 = pd.read_csv('output/BPIC2017/final/tfcl/prediction_results2.csv')
tfcl3 = pd.read_csv('output/BPIC2017/final/tfcl/prediction_results3.csv')
tfcl4 = pd.read_csv('output/BPIC2017/final/tfcl/prediction_results4.csv')

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

x1 = calculate_percentage_of_ones(tfcl_0)
y1 = range(500, len(x1)+500)
r1 = calculate_percentage_of_ones(tfcl_1)
r2 = calculate_percentage_of_ones(tfcl_2)
r3 = calculate_percentage_of_ones(tfcl_3)
r4 = calculate_percentage_of_ones(tfcl_4)
acc = [sum(x)/len(x) for x in [x1, r1,r2, r3, r4]] 
avg_acc_lastdrift = np.mean(acc)
print(f'Accuracies of tfcl: {acc}')
print(f'Average accuracy of tfcl: {avg_acc_lastdrift}')

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

x3 = dual0.iloc[:,0]
y3 = dual0.iloc[:,1]
r1 = dual1.iloc[:,1]
r2 = dual2.iloc[:,1]
r3 = dual3.iloc[:,1]
r4 = dual4.iloc[:,1]
acc = [sum(x)/len(x) for x in [y3, r1,r2, r3, r4]] 
avg_acc_lastdrift = np.mean(acc)
print(f'Accuracies of dualprompt: {acc}')
print(f'Average accuracy of dualprompt: {avg_acc_lastdrift}')

x5 = calculate_percentage_of_ones(drift0)
y5 = range(500, len(x5)+500)
r1 = calculate_percentage_of_ones(drift1)
r2 = calculate_percentage_of_ones(drift2)
r3 = calculate_percentage_of_ones(drift3)
r4 = calculate_percentage_of_ones(drift4)
acc = [sum(x)/len(x) for x in [x5, r1,r2, r3, r4]] 
avg_acc_lastdrift = np.mean(acc)
print(f'Accuracies of last drifts: {acc}')
print(f'Average accuracy of last drifts: {avg_acc_lastdrift}')

x6 = calculate_percentage_of_ones(land0)
y6 = range(500, len(x6)+500)
r1 = calculate_percentage_of_ones(land1)
r2 = calculate_percentage_of_ones(land2)
r3 = calculate_percentage_of_ones(land3)
r4 = calculate_percentage_of_ones(land4)
acc = [sum(x)/len(x) for x in [x6, r1,r2, r3, r4]] 
avg_acc_lastdrift = np.mean(acc)
print(f'Accuracies of landmark: {acc}')
print(f'Average accuracy of landmark: {avg_acc_lastdrift}')

plt.figure(figsize=(20,5))
plt.plot(y5, x5, linestyle='-', linewidth=1, color = 'orange', label = 'Last drift')  
plt.plot(y6, x6, linestyle='-', linewidth=1, color = 'yellow', label = 'Landmark')  
plt.plot(y4, x4, linestyle='-', linewidth=1, color = 'pink', label = 'GAN')
plt.plot(y, x, linestyle='-', linewidth=1, color = 'red', label = 'DynaTrainCDD')
plt.plot(y1, x1, linestyle='-', linewidth=1, color = 'green', label = 'tfcl') 
plt.plot(x3, y3*100, linestyle='-', linewidth=1, color = 'blue', label = 'DualPrompt')  
   
c = [0, len(x)+500]

plt.axvspan(c[0], c[1], alpha=0.3, color = 'blue', label = 'Task 1')

plt.xticks(range(0, len(x), 10000))
plt.xlabel('Event')
plt.ylabel('Accuracy')
plt.title('BPIC2017')
plt.grid(True)  # Add grid lines (optional)
plt.legend()
plt.show()

avg_acc = sum(x5)/len(x5) #last drift
print(f'Last drift: {avg_acc}')
avg_acc = sum(x6)/len(x6) #landmark
print(f'Landmark: {avg_acc}')
avg_acc = sum(y3)/len(y3) #dualprompt
print(f'Dualprompt: {avg_acc*100}')
avg_acc = sum(x4)/len(x4)
print(f'GAN: {avg_acc}')
avg_acc = sum(x)/len(x) #dynatraincdd
print(f'DynaTrainCDD: {avg_acc}')
avg_acc = sum(x1)/len(x1) 
print(f'tfcl: {avg_acc}')
