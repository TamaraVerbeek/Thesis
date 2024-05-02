# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:01:46 2024

@author: tamar
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dyna = pd.read_csv('output/random_dataset/final/dynatraincdd/output_DynaTrainCDD_randomtasks_0.csv')
dyna1 = pd.read_csv('output/random_dataset/final/dynatraincdd/output_DynaTrainCDD_randomtasks_1.csv')
dyna2 = pd.read_csv('output/random_dataset/final/dynatraincdd/output_DynaTrainCDD_randomtasks_2.csv')
dyna3 = pd.read_csv('output/random_dataset/final/dynatraincdd/output_DynaTrainCDD_randomtasks_3.csv')
dyna4 = pd.read_csv('output/random_dataset/final/dynatraincdd/output_DynaTrainCDD_randomtasks_4.csv')

gan = pd.read_csv('output/random_dataset/final/gan/output_GAN_randomtasks_0.csv')
gan_1 = pd.read_csv('output/random_dataset/final/gan/output_GAN_randomtasks_1.csv')
gan_2 = pd.read_csv('output/random_dataset/final/gan/output_GAN_randomtasks_2.csv')
gan_3 = pd.read_csv('output/random_dataset/final/gan/output_GAN_randomtasks_3.csv')
gan_4 = pd.read_csv('output/random_dataset/final/gan/output_GAN_randomtasks_4.csv')


dual0 = pd.read_csv('output/random_dataset/final/dualprompt/output250_RandomTasks.csv_0.csv')
dual1 = pd.read_csv('output/random_dataset/final/dualprompt/output_dualprompt_RandomTasks.csv_1.csv')
dual2 = pd.read_csv('output/random_dataset/final/dualprompt/output_dualprompt_RandomTasks.csv_2.csv')
dual3 = pd.read_csv('output/random_dataset/final/dualprompt/output_dualprompt_RandomTasks.csv_3.csv')
dual4 = pd.read_csv('output/random_dataset/final/dualprompt/output_dualprompt_RandomTasks.csv_4.csv')

lastdrift0 = pd.read_csv('output/random_dataset/final/lastdrift/output_static_lastdrift_random_0.csv')
lastdrift1 = pd.read_csv('output/random_dataset/final/lastdrift/output_static_lastdrift_random_1.csv')
lastdrift2 = pd.read_csv('output/random_dataset/final/lastdrift/output_static_lastdrift_random_2.csv')
lastdrift3 = pd.read_csv('output/random_dataset/final/lastdrift/output_static_lastdrift_random_3.csv')
lastdrift4 = pd.read_csv('output/random_dataset/final/lastdrift/output_static_lastdrift_random_4.csv')

static0 = pd.read_csv('output/random_dataset/final/landmark/output_static_woppie_0.csv')
static1 = pd.read_csv('output/random_dataset/final/landmark/output_static_woppie_1.csv')
static2 = pd.read_csv('output/random_dataset/final/landmark/output_static_woppie_2.csv')
static3 = pd.read_csv('output/random_dataset/final/landmark/output_static_woppie_3.csv')
static4 = pd.read_csv('output/random_dataset/final/landmark/output_static_woppie_4.csv')

tfcl0 = pd.read_csv('output/random_dataset/final/tfcl/prediction_results.csv')
tfcl1 = pd.read_csv('output/random_dataset/final/tfcl/prediction_results1.csv')
tfcl2 = pd.read_csv('output/random_dataset/final/tfcl/prediction_results2.csv')
tfcl3 = pd.read_csv('output/random_dataset/final/tfcl/prediction_results3.csv')
tfcl4 = pd.read_csv('output/random_dataset/final/tfcl/prediction_results4.csv')

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
        
tfcl_0 = get0and1_other(tfcl0)
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


x1 = calculate_percentage_of_ones(list(gan['0']))
y1 = range(500, len(x1)+500)
r1 = calculate_percentage_of_ones(gan_1['0'])
r2 = calculate_percentage_of_ones(gan_2['0'])
r3 = calculate_percentage_of_ones(gan_3['0'])
r4 = calculate_percentage_of_ones(gan_4['0'])
acc = [sum(x)/len(x) for x in [x1, r1,r2, r3, r4]] 
avg_acc_gan = np.mean(acc)
print(f'Accuracies of GAN: {acc}')
print(f'Average accuracy of GAN: {avg_acc_gan}')

x2 = dual0.iloc[:,0]
y2 = dual0.iloc[:,1]
dual_1 = dual1.iloc[:,1]
dual_2 = dual2.iloc[:,1]
dual_3 = dual3.iloc[:,1]
dual_4 = dual4.iloc[:,1]
acc = [sum(x)/len(x) for x in [y2, dual_1, dual_2, dual_3, dual_4]] 
avg_acc_landmark = np.mean(acc)
print(f'Accuracies of dualprompt: {acc}')
print(f'Average accuracy of dualprompt: {avg_acc_landmark}')

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

x4 = calculate_percentage_of_ones(land0)
y4 = range(500, len(x4)+500)
r1 = calculate_percentage_of_ones(land1)
r2 = calculate_percentage_of_ones(land2)
r3 = calculate_percentage_of_ones(land3)
r4 = calculate_percentage_of_ones(land4)
acc = [sum(x)/len(x) for x in [x4, r1,r2, r3, r4]] 
avg_acc_lastdrift = np.mean(acc)
print(f'Accuracies of landmark: {acc}')
print(f'Average accuracy of landmark: {avg_acc_lastdrift}')

#x5 = data5.iloc[:,0]
#y5 = data5.iloc[:,1]

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

#x6 = data6.iloc[:,0]
#y6 = data6.iloc[:,1]

#x8 = data7.iloc[:,0]
#y8 = data7.iloc[:,1]

concept_drift = [2724, 5756, 9074, 12025, 15297, 18042, 20680, 23385, 26816, 30068, 33105, 36101, 38786, 41614, 44979, 47623, 50125, 52627, 55129, 57631, 60133, 63493, 66198, 69256, 72060, 75121, 77904]
plt.figure(figsize=(20,5))
for index in concept_drift:
    plt.axvline(x=index, color='red', linestyle='--', linewidth=1)  # Adjust color, linestyle, and linewidth as needed
c = [0, 2724, 5756, 9074, 12025, 15297, 18042, 20680, 23385, 26816, 30068, 33105, 36101, 38786, 41614, 44979, 47623, 50125, 52627, 55129, 57631, 60133, 63493, 66198, 69256, 72060, 75121, 77904, len(x)+500] #combi
plt.axvspan(c[0], c[1], alpha=0.3, color = 'blue', label = 'Task 1')
plt.axvspan(c[7], c[8], alpha=0.3, color = 'blue')
plt.axvspan(c[15], c[16], alpha=0.3, color = 'blue')
plt.axvspan(c[22], c[23], alpha=0.3, color = 'blue')

plt.axvspan(c[1], c[2], alpha=0.3, color = 'green', label = 'Task 2')
plt.axvspan(c[11], c[12], alpha=0.3, color = 'green')
plt.axvspan(c[16], c[17], alpha=0.3, color = 'green')
plt.axvspan(c[25], c[26], alpha=0.3, color = 'green')


plt.axvspan(c[2], c[3], alpha=0.3, color = 'purple', label = 'Task 3')
plt.axvspan(c[8], c[9], alpha=0.3, color = 'purple')
plt.axvspan(c[14], c[15], alpha=0.3, color = 'purple')
plt.axvspan(c[27], c[28], alpha=0.3, color = 'purple')


plt.axvspan(c[3], c[4], alpha=0.3, color = 'pink', label = 'Task 4')
plt.axvspan(c[10], c[11], alpha=0.3, color = 'pink')
plt.axvspan(c[20], c[21], alpha=0.3, color = 'pink')
plt.axvspan(c[23], c[24], alpha=0.3, color = 'pink')


plt.axvspan(c[4], c[5], alpha=0.3, color = 'orange', label = 'Task 5')
plt.axvspan(c[9], c[10], alpha=0.3, color = 'orange')
plt.axvspan(c[18], c[19], alpha=0.3, color = 'orange')
plt.axvspan(c[21], c[22], alpha=0.3, color = 'orange')


plt.axvspan(c[5], c[6], alpha=0.3, color = 'yellow', label = 'Task 6')
plt.axvspan(c[13], c[14], alpha=0.3, color = 'yellow')
plt.axvspan(c[19], c[20], alpha=0.3, color = 'yellow')
plt.axvspan(c[26], c[27], alpha=0.3, color = 'yellow')


plt.axvspan(c[6], c[7], alpha=0.3, color = 'red', label = 'Task 7')
plt.axvspan(c[12], c[13], alpha=0.3, color = 'red')
plt.axvspan(c[17], c[18], alpha=0.3, color = 'red')
plt.axvspan(c[24], c[25], alpha=0.3, color = 'red')



# Plotting the line plot
plt.plot(y, x, linestyle='-', linewidth=1, color = 'red', label = 'DynaTrainCDD')
plt.plot(y3, x3, linestyle='-', linewidth=1, color = 'orange', label='Last drift')
plt.plot(y4, x4, linestyle='-', linewidth=1, color = 'yellow', label = 'Landmark')
plt.plot(y1, x1, linestyle='-', linewidth=1, color = 'purple', label='GAN')
#plt.plot(x5, y5*100,linestyle='-', linewidth=1, color = 'blue', label='Dualprompt')
#plt.plot(x6, y6*100,linestyle='-', linewidth=1, color = 'yellow', label='E-Prompt')
#plt.plot(x8, y8*100,linestyle='-', linewidth=1, color = 'red', label='G-Prompt')
plt.plot(y7, x7, linestyle='-', linewidth=1, color = 'green', label = 'tfcl') 
plt.plot(x2, y2*100,linestyle='-', linewidth=1, color = 'blue', label='Dualprompt')


for index in concept_drift:
    plt.axvline(x=index, color='red', linestyle='--', linewidth=1)  # Adjust color, linestyle, and linewidth as needed
    
plt.xticks(range(0, len(x), 5000))
plt.xlabel('Event')
plt.ylabel('Accuracy')
plt.title('RandomTasks dataset')
plt.grid(True)  # Add grid lines (optional)
plt.legend(loc='lower right')
plt.show()

avg_acc = sum(x)/len(x)
print(f'DynaTrainCDD: {avg_acc}')
avg_acc = sum(x1)/len(x1)
print(f'GAN: {avg_acc}')
#avg_acc = sum(y5) / len(y5)
#print(f'DualPrompt: {avg_acc*100}')
avg_acc = sum(x3)/len(x3) 
print(f'Static last drift: {avg_acc}')
avg_acc = sum(x4)/len(x4) 
print(f'Landmark: {avg_acc}')
avg_acc = sum(y2)/len(y2)
print(f'Dualprompt: {avg_acc*100}')
#avg_acc = sum(y6)/len(y6)
#print(f'E-prompt: {avg_acc*100}')
avg_acc = sum(x7)/len(x7) 
print(f'tfcl: {avg_acc}')
#avg_acc = sum(y8)/len(y8)
#print(f'G-prompt: {avg_acc*100}')