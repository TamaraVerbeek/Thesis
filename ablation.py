# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:22:58 2024

@author: tamar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dual0 = pd.read_csv('output/bigloop_dataset/final/dualprompt/output_dualprompt_RecurrentTasks.csv_0.csv')
dual1 = pd.read_csv('output/bigloop_dataset/final/dualprompt/output_dualprompt_RecurrentTasks.csv_1.csv')
dual2 = pd.read_csv('output/bigloop_dataset/final/dualprompt/output_dualprompt_RecurrentTasks.csv_2.csv')
dual3 = pd.read_csv('output/bigloop_dataset/final/dualprompt/output_dualprompt_RecurrentTasks.csv_3.csv')
dual4 = pd.read_csv('output/bigloop_dataset/final/dualprompt/output_dualprompt_RecurrentTasks.csv_4.csv')

eprompt0 = pd.read_csv('output/bigloop_dataset/final results dualprompt/output_eprompt_loop_0.csv')
eprompt1 = pd.read_csv('output/bigloop_dataset/final results dualprompt/output_eprompt_loop_1.csv')
eprompt2 = pd.read_csv('output/bigloop_dataset/final results dualprompt/output_eprompt_loop_2.csv')
eprompt3 = pd.read_csv('output/bigloop_dataset/final results dualprompt/output_eprompt_loop_3.csv')
eprompt4 = pd.read_csv('output/bigloop_dataset/final results dualprompt/output_eprompt_loop_4.csv')

gprompt0 = pd.read_csv('output/bigloop_dataset/final results dualprompt/output_gprompt_loop_0.csv')
gprompt1 = pd.read_csv('output/bigloop_dataset/final results dualprompt/output_gprompt_loop_1.csv')
gprompt2 = pd.read_csv('output/bigloop_dataset/final results dualprompt/output_gprompt_loop_2.csv')
gprompt3 = pd.read_csv('output/bigloop_dataset/final results dualprompt/output_gprompt_loop_3.csv')
gprompt4 = pd.read_csv('output/bigloop_dataset/final results dualprompt/output_gprompt_loop_4.csv')

noprompt0 = pd.read_csv('output/bigloop_dataset/final results dualprompt/output_noprompt_loop_0.csv')
noprompt1 = pd.read_csv('output/bigloop_dataset/final results dualprompt/output_noprompt_loop_1.csv')
noprompt2 = pd.read_csv('output/bigloop_dataset/final results dualprompt/output_noprompt_loop_2.csv')
noprompt3 = pd.read_csv('output/bigloop_dataset/final results dualprompt/output_noprompt_loop_3.csv')
noprompt4 = pd.read_csv('output/bigloop_dataset/final results dualprompt/output_noprompt_loop_4.csv')

dual_00 = dual0.iloc[:,1]
y0 = dual0.iloc[:,0]
dual_1 = dual1.iloc[:,1]
dual_2 = dual2.iloc[:,1]
dual_3 = dual3.iloc[:,1]
dual_4 = dual4.iloc[:,1]
acc = [sum(x)/len(x) for x in [dual_00, dual_1, dual_2, dual_3, dual_4]] 
avg_acc_landmark = np.mean(acc)
print(f'Accuracies of dualprompt: {acc}')
print(f'Average accuracy of dualprompt: {avg_acc_landmark}')

dual_01 = eprompt0.iloc[:,1]
y1 = eprompt0.iloc[:,0]
dual_1 = eprompt1.iloc[:,1]
dual_2 = eprompt2.iloc[:,1]
dual_3 = eprompt3.iloc[:,1]
dual_4 = eprompt4.iloc[:,1]
acc = [sum(x)/len(x) for x in [dual_01, dual_1, dual_2, dual_3, dual_4]] 
avg_acc_landmark = np.mean(acc)
print(f'Accuracies of E-prompt: {acc}')
print(f'Average accuracy of E-prompt: {avg_acc_landmark}')

dual_02 = gprompt0.iloc[:,1]
y2 = gprompt0.iloc[:,0]
dual_1 = gprompt1.iloc[:,1]
dual_2 = gprompt2.iloc[:,1]
dual_3 = gprompt3.iloc[:,1]
dual_4 = gprompt4.iloc[:,1]
acc = [sum(x)/len(x) for x in [dual_02, dual_1, dual_2, dual_3, dual_4]] 
avg_acc_landmark = np.mean(acc)
print(f'Accuracies of G-prompt: {acc}')
print(f'Average accuracy of G-prompt: {avg_acc_landmark}')

dual_03 = noprompt0.iloc[:,1]
y3 = noprompt0.iloc[:,0]
dual_1 = noprompt1.iloc[:,1]
dual_2 = noprompt2.iloc[:,1]
dual_3 = noprompt3.iloc[:,1]
dual_4 = noprompt4.iloc[:,1]
acc = [sum(x)/len(x) for x in [dual_03, dual_1, dual_2, dual_3, dual_4]] 
avg_acc_landmark = np.mean(acc)
print(f'Accuracies of no prompt: {acc}')
print(f'Average accuracy of no prompt: {avg_acc_landmark}')

concept_drift =  [2725, 5757, 9075, 12071, 15502, 18207, 21268, 24633, 27758, 31178, 33822, 36838, 40221, 42722, 45223, 47725, 50563, 53523, 56323, 59190, 61790, 64512, 67380, 70182, 73124, 75720, 78222, 80724, 83226, 85728, 88230, 90732, 93234, 95736, 98238] #combi
plt.figure(figsize=(20,5))
plt.plot(y1, dual_01*100, linestyle='-', linewidth=1, color = 'green', label = 'E-Prompt')  
plt.plot(y2, dual_02*100, linestyle='-', linewidth=1, color = 'yellow', label = 'G-Prompt')  
plt.plot(y3, dual_03*100, linestyle='-', linewidth=1, color = 'red', label = 'No Prompt')  
plt.plot(y0, dual_00*100, linestyle='-', linewidth=1, color = 'blue', label = 'Dualprompt')  

for index in concept_drift:
    plt.axvline(x=index, color='red', linestyle='--', linewidth=1)  # Adjust color, linestyle, and linewidth as needed
c = [500, 2725, 5757, 9075, 12071, 15502, 18207, 21268, 24633, 27758, 31178, 33822, 36838, 40221, 42722, 45223, 47725, 50563, 53523, 56323, 59190, 61790, 64512, 67380, 70182, 73124, 75720, 78222, 80724, 83226, 85728, 88230, 90732, 93234, 95736, 98238, len(dual_00)+500] #combi

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

plt.xticks(range(0, len(dual_00), 5000))
plt.xlabel('Event')
plt.ylabel('Accuracy')
plt.title('RecurrentTasks dataset')
plt.grid(True)  # Add grid lines (optional)
plt.legend()
plt.show()
