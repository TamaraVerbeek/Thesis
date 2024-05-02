# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:59:16 2024

@author: tamar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

def calculate_percentage_of_ones(data):
    window_size = 250
    result = []

    for i in range(len(data)):
        start_index = max(0, i - window_size + 1)
        window = data[start_index:i+1]
        percentage_of_ones = sum(window) / len(window) * 100
        result.append(percentage_of_ones)

    return result


def get_accs(data, concept_drift, dataset):
    if 'bigloop' in dataset:
        acc_t1_1 = np.mean(data.iloc[0:concept_drift[0] ])
        acc_t1_2 = np.mean(data.iloc[concept_drift[4]:concept_drift[5] ])
        acc_t1_3 = np.mean(data.iloc[concept_drift[9]:concept_drift[10] ])
        acc_t1_4 = np.mean(data.iloc[concept_drift[14]:concept_drift[15] ])
        acc_t1_5 = np.mean(data.iloc[concept_drift[19]:concept_drift[20] ])
        acc_t1_6 = np.mean(data.iloc[concept_drift[24]:concept_drift[25] ])
        acc_t1_7 = np.mean(data.iloc[concept_drift[29]:concept_drift[30] ])
        acc_t1_8 = np.mean(data.iloc[concept_drift[34]:len(data) ])
        acc_t1_9 = acc_t1_2#np.mean(data.iloc[concept_drift[20]:concept_drift[21] ])
        acc_t1_10 = acc_t1_2#np.mean(data.iloc[concept_drift[22]:concept_drift[23] ])
        acc_t1_11 = acc_t1_2#np.mean(data.iloc[concept_drift[25]:concept_drift[26] ])
        acc_t1_12 = acc_t1_2#np.mean(data.iloc[concept_drift[27]:concept_drift[28] ])
        acc_t1_13 = acc_t1_2#np.mean(data.iloc[concept_drift[30]:concept_drift[31] ])
        acc_t1_14 = acc_t1_2#np.mean(data.iloc[concept_drift[32]:concept_drift[33] ])
        acc_t1_15 = acc_t1_2#np.mean(data.iloc[concept_drift[32]:concept_drift[33] ])

        acc_t2_1 = np.mean(data.iloc[concept_drift[0]:concept_drift[1] ])
        acc_t2_2 = np.mean(data.iloc[concept_drift[2]:concept_drift[3] ])
        acc_t2_3 = np.mean(data.iloc[concept_drift[5]:concept_drift[6] ])
        acc_t2_4 = np.mean(data.iloc[concept_drift[7]:concept_drift[8] ])
        acc_t2_5 = np.mean(data.iloc[concept_drift[10]:concept_drift[11] ])
        acc_t2_6 = np.mean(data.iloc[concept_drift[12]:concept_drift[13] ])
        acc_t2_7 = np.mean(data.iloc[concept_drift[15]:concept_drift[16] ])
        acc_t2_8 = np.mean(data.iloc[concept_drift[17]:concept_drift[18] ])
        acc_t2_9 = np.mean(data.iloc[concept_drift[20]:concept_drift[21] ])
        acc_t2_10 = np.mean(data.iloc[concept_drift[22]:concept_drift[23] ])
        acc_t2_11 = np.mean(data.iloc[concept_drift[25]:concept_drift[26] ])
        acc_t2_12 = np.mean(data.iloc[concept_drift[27]:concept_drift[28] ])
        acc_t2_13 = np.mean(data.iloc[concept_drift[30]:concept_drift[31] ])
        acc_t2_14 = np.mean(data.iloc[concept_drift[32]:concept_drift[33] ])
        
        acc_t3_1 = np.mean(data.iloc[concept_drift[1]:concept_drift[2] ])
        acc_t3_2 = np.mean(data.iloc[concept_drift[3]:concept_drift[4] ])
        acc_t3_3 = np.mean(data.iloc[concept_drift[6]:concept_drift[7] ])
        acc_t3_4 = np.mean(data.iloc[concept_drift[8]:concept_drift[9] ])
        acc_t3_5 = np.mean(data.iloc[concept_drift[11]:concept_drift[12] ])
        acc_t3_6 = np.mean(data.iloc[concept_drift[13]:concept_drift[14] ])
        acc_t3_7 = np.mean(data.iloc[concept_drift[16]:concept_drift[17] ])
        acc_t3_8 = np.mean(data.iloc[concept_drift[18]:concept_drift[19] ])
        acc_t3_9 = np.mean(data.iloc[concept_drift[21]:concept_drift[22] ])
        acc_t3_10 = np.mean(data.iloc[concept_drift[23]:concept_drift[24] ])
        acc_t3_11 = np.mean(data.iloc[concept_drift[26]:concept_drift[27] ])
        acc_t3_12 = np.mean(data.iloc[concept_drift[28]:concept_drift[29] ])
        acc_t3_13 = np.mean(data.iloc[concept_drift[31]:concept_drift[32] ])
        acc_t3_14 = np.mean(data.iloc[concept_drift[33]:concept_drift[34] ])
        
        first_time = np.array([acc_t1_2, acc_t2_1 , acc_t3_1])
        other_times = np.array([
            [acc_t1_15, acc_t2_14, acc_t3_14],   # Second time
            [acc_t1_14, acc_t2_13, acc_t3_13],   # Second time
            [acc_t1_13, acc_t2_12, acc_t3_12],   # Third time
            [acc_t1_12, acc_t2_11, acc_t3_11],   # Fourth time
            [acc_t1_11, acc_t2_10, acc_t3_10],   # Second time
            [acc_t1_10, acc_t2_9, acc_t3_9],   # Second time
            [acc_t1_9, acc_t2_8, acc_t3_8],   # Third time
            [acc_t1_8, acc_t2_7, acc_t3_7],   # Fourth time
            [acc_t1_7, acc_t2_6, acc_t3_6],   # Second time
            [acc_t1_6, acc_t2_5, acc_t3_5],   # Third time
            [acc_t1_5, acc_t2_4, acc_t3_4],   # Fourth time
            [acc_t1_4, acc_t2_3, acc_t3_3],   # Second time
            [acc_t1_3, acc_t2_2, acc_t3_2]    # Fourth time
        ])
        all_times = np.vstack([first_time, other_times])
        differences = other_times - first_time
        print(all_times)
    elif 'infrequent' in dataset:
        acc_t1_1 = np.mean(data.iloc[0:concept_drift[0] ])
        acc_t1_2 = np.mean(data.iloc[concept_drift[4]:concept_drift[5] ])
        acc_t1_3 = np.mean(data.iloc[concept_drift[7]:concept_drift[8] ])
        acc_t1_4 = np.mean(data.iloc[concept_drift[9]:len(data)])
        acc_t1_5 = acc_t1_2
        
        acc_t2_1 = np.mean(data.iloc[concept_drift[0]:concept_drift[1] ])
        acc_t2_2 = np.mean(data.iloc[concept_drift[6]:concept_drift[7] ])
        acc_t2_3 = np.mean(data.iloc[concept_drift[8]:concept_drift[9]])
        acc_t2_4 = acc_t2_1
        
        acc_t3_1 = np.mean(data.iloc[concept_drift[1]:concept_drift[2] ])
        acc_t3_2 = np.mean(data.iloc[concept_drift[3]:concept_drift[4] ])
        acc_t3_3 = np.mean(data.iloc[concept_drift[5]:concept_drift[6] ])
        acc_t3_4 = acc_t3_1
        
        acc_t4_1 = np.mean(data.iloc[concept_drift[2]:concept_drift[3] ])
        acc_t4_2 = acc_t4_1
        acc_t4_3 = acc_t4_1
        acc_t4_4 = acc_t4_1

        first_time = np.array([acc_t1_2, acc_t2_1 , acc_t3_1, acc_t4_1])
        other_times = np.array([
            [acc_t1_5 , acc_t2_4 , acc_t3_4, acc_t4_4],   # Fourth time
            [acc_t1_4 ,acc_t2_3 ,acc_t3_3, acc_t4_3],  # Second time
            [acc_t1_3 , acc_t2_2 , acc_t3_2, acc_t4_2],   # Fourth time
            ])
        all_times = np.vstack([first_time, other_times])
        differences = other_times - first_time
        print(all_times)
    elif 'BPIC2015' in dataset:
        acc_t1_1 = np.mean(data.iloc[0:concept_drift[0] ])
        acc_t1_2 = np.mean(data.iloc[concept_drift[3]:concept_drift[4] ])
        acc_t1_3 = np.mean(data.iloc[concept_drift[7]:concept_drift[8] ])
        acc_t1_4 = np.mean(data.iloc[concept_drift[11]:concept_drift[12]])
        acc_t1_5 = acc_t1_2
        
        acc_t2_1 = np.mean(data.iloc[concept_drift[0]:concept_drift[1] ])
        acc_t2_2 = np.mean(data.iloc[concept_drift[4]:concept_drift[5] ])
        acc_t2_3 = np.mean(data.iloc[concept_drift[8]:concept_drift[9]])
        acc_t2_4 = np.mean(data.iloc[concept_drift[12]:concept_drift[13]])
        
        acc_t3_1 = np.mean(data.iloc[concept_drift[1]:concept_drift[2] ])
        acc_t3_2 = np.mean(data.iloc[concept_drift[5]:concept_drift[6] ])
        acc_t3_3 = np.mean(data.iloc[concept_drift[9]:concept_drift[10] ])
        acc_t3_4 = np.mean(data.iloc[concept_drift[13]:concept_drift[14] ])
        
        acc_t4_1 = np.mean(data.iloc[concept_drift[2]:concept_drift[3] ])
        acc_t4_2 = np.mean(data.iloc[concept_drift[6]:concept_drift[7] ])
        acc_t4_3 = np.mean(data.iloc[concept_drift[10]:concept_drift[11] ])
        acc_t4_4 = np.mean(data.iloc[concept_drift[14]:len(data)])

        first_time = np.array([acc_t1_2, acc_t2_1 , acc_t3_1, acc_t4_1])-2
        other_times = np.array([
            [acc_t1_5 , acc_t2_4 , acc_t3_4, acc_t4_4],   # Fourth time
            [acc_t1_4 ,acc_t2_3 ,acc_t3_3, acc_t4_3],  # Second time
            [acc_t1_3 , acc_t2_2 , acc_t3_2, acc_t4_2],   # Fourth time
            ])-2
        all_times = np.vstack([first_time, other_times])
        differences = other_times - first_time
        print(all_times)
    elif 'random' in dataset:
        acc_t1 = np.mean([np.mean(data.iloc[0:concept_drift[0] ]), np.mean(data.iloc[concept_drift[6]:concept_drift[7] ]), np.mean(data.iloc[concept_drift[14]:concept_drift[15] ]), np.mean(data.iloc[concept_drift[21]:concept_drift[22] ])])
        acc_t1_1 = np.mean(data.iloc[0:concept_drift[0] ])
        acc_t1_2 = np.mean(data.iloc[concept_drift[6]:concept_drift[7] ])
        acc_t1_3 = np.mean(data.iloc[concept_drift[14]:concept_drift[15] ])
        acc_t1_4 = np.mean(data.iloc[concept_drift[21]:concept_drift[22] ])
        print('task 1')
        print(f'average of all {acc_t1}')
        print(acc_t1_1)
        print(acc_t1_2)
        print(acc_t1_3)
        print(acc_t1_4)
        
        acc_t2 = np.mean([np.mean(data.iloc[concept_drift[0]:concept_drift[1] ]), np.mean(data.iloc[concept_drift[10]:concept_drift[11] ]), np.mean(data.iloc[concept_drift[15]:concept_drift[16] ]), np.mean(data.iloc[concept_drift[24]:concept_drift[25] ])])
        acc_t2_1 = np.mean(data.iloc[concept_drift[0]:concept_drift[1] ])
        acc_t2_2 = np.mean(data.iloc[concept_drift[10]:concept_drift[11] ])
        acc_t2_3 = np.mean(data.iloc[concept_drift[15]:concept_drift[16] ])
        acc_t2_4 = np.mean(data.iloc[concept_drift[24]:concept_drift[25] ])
        print('task 2')
        print(f'average of all {acc_t2}')
        print(acc_t2_1)
        print(acc_t2_2)
        print(acc_t2_3)
        print(acc_t2_4)
        
        acc_t3 = np.mean([np.mean(data.iloc[concept_drift[1]:concept_drift[2] ]), np.mean(data.iloc[concept_drift[7]:concept_drift[8] ]), np.mean(data.iloc[concept_drift[13]:concept_drift[14] ]), np.mean(data.iloc[concept_drift[26]:len(data) ])])
        acc_t3_1 = np.mean(data.iloc[concept_drift[1]:concept_drift[2] ])
        acc_t3_2 = np.mean(data.iloc[concept_drift[7]:concept_drift[8] ])
        acc_t3_3 = np.mean(data.iloc[concept_drift[13]:concept_drift[14] ])
        acc_t3_4 = np.mean(data.iloc[concept_drift[26]:len(data) ])
        print('task 3')
        print(f'average of all {acc_t3}')
        print(acc_t3_1)
        print(acc_t3_2)
        print(acc_t3_3)
        print(acc_t3_4)
        
        acc_t4 = np.mean([np.mean(data.iloc[concept_drift[2]:concept_drift[3] ]), np.mean(data.iloc[concept_drift[9]:concept_drift[10] ]), np.mean(data.iloc[concept_drift[19]:concept_drift[20] ]), np.mean(data.iloc[concept_drift[22]:concept_drift[23] ])])
        acc_t4_1 = np.mean(data.iloc[concept_drift[2]:concept_drift[3] ])
        acc_t4_2 = np.mean(data.iloc[concept_drift[9]:concept_drift[10] ])
        acc_t4_3 = np.mean(data.iloc[concept_drift[19]:concept_drift[20] ])
        acc_t4_4 = np.mean(data.iloc[concept_drift[22]:concept_drift[23] ])
        print('task 4')
        print(f'average of all {acc_t4}')
        print(acc_t4_1)
        print(acc_t4_2)
        print(acc_t4_3)
        print(acc_t4_4)
        
        acc_t5 = np.mean([np.mean(data.iloc[concept_drift[3]:concept_drift[4] ]), np.mean(data.iloc[concept_drift[8]:concept_drift[9] ]), np.mean(data.iloc[concept_drift[17]:concept_drift[18] ]), np.mean(data.iloc[concept_drift[20]:concept_drift[21] ])])
        acc_t5_1 = np.mean(data.iloc[concept_drift[3]:concept_drift[4] ])
        acc_t5_2 = np.mean(data.iloc[concept_drift[8]:concept_drift[9] ])
        acc_t5_3 = np.mean(data.iloc[concept_drift[17]:concept_drift[18] ])
        acc_t5_4 = np.mean(data.iloc[concept_drift[20]:concept_drift[21] ])
        print('task 5')
        print(f'average of all {acc_t5}')
        print(acc_t5_1)
        print(acc_t5_2)
        print(acc_t5_3)
        print(acc_t5_4)
        
        acc_t6 = np.mean([np.mean(data.iloc[concept_drift[4]:concept_drift[5] ]), np.mean(data.iloc[concept_drift[12]:concept_drift[13] ]), np.mean(data.iloc[concept_drift[18]:concept_drift[19] ]), np.mean(data.iloc[concept_drift[25]:concept_drift[26] ])])
        acc_t6_1 = np.mean(data.iloc[concept_drift[4]:concept_drift[5] ])
        acc_t6_2 = np.mean(data.iloc[concept_drift[12]:concept_drift[13] ])
        acc_t6_3 = np.mean(data.iloc[concept_drift[18]:concept_drift[19] ])
        acc_t6_4 = np.mean(data.iloc[concept_drift[25]:concept_drift[26] ])
        print('task 6')
        print(f'average of all {acc_t6}')
        print(acc_t6_1)
        print(acc_t6_2)
        print(acc_t6_3)
        print(acc_t6_4)
        
        acc_t7 = np.mean([np.mean(data.iloc[concept_drift[5]:concept_drift[6] ]), np.mean(data.iloc[concept_drift[11]:concept_drift[12] ]), np.mean(data.iloc[concept_drift[16]:concept_drift[17] ]), np.mean(data.iloc[concept_drift[23]:concept_drift[24] ])])
        acc_t7_1 = np.mean(data.iloc[concept_drift[5]:concept_drift[6] ])
        acc_t7_2 = np.mean(data.iloc[concept_drift[11]:concept_drift[12] ])
        acc_t7_3 = np.mean(data.iloc[concept_drift[16]:concept_drift[17] ])
        acc_t7_4 = np.mean(data.iloc[concept_drift[23]:concept_drift[24] ])
        print('task 7')
        print(f'average of all {acc_t7}')
        print(acc_t7_1)
        print(acc_t7_2)
        print(acc_t7_3)
        print(acc_t7_4)
    
        # Sample accuracy values for the first time and subsequent three times
        first_time = np.array([acc_t1_2, acc_t2_1 , acc_t3_1 , acc_t4_1 , acc_t5_1 , acc_t6_1 , acc_t7_1 ])
        other_times = np.array([
            [acc_t1_2 , acc_t2_4 , acc_t3_4 , acc_t4_4 , acc_t5_4 , acc_t6_4 , acc_t7_4 ],   # Fourth time
            [acc_t1_4 , acc_t2_3 , acc_t3_3 , acc_t4_3 , acc_t5_3 , acc_t6_3 , acc_t7_3 ],  # Third time
            [acc_t1_3 ,acc_t2_2 ,acc_t3_2 ,acc_t4_2 , acc_t5_2 , acc_t6_2 , acc_t7_2 ]  # Second time
        ])
        all_times = np.vstack([first_time, other_times])
        differences = other_times - first_time
        print(differences.shape)
        print(first_time.shape)
        print(all_times)
    # Compute the differences between the first time and other times
    return differences, first_time, all_times
    

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titles = ['DualPrompt','DynaTrainCDD',  'tfcl', 'GAN', 'Landmark', 'Last drift']
#datasets = ['output/random_dataset/final/dualprompt/output_dualprompt_RandomTasks.csv_3.csv', 'output/random_dataset/final/dynatraincdd/output_DynaTrainCDD_randomtasks_4.csv', 'output/random_dataset/final/tfcl/prediction_results4.csv', 'output/random_dataset/final/gan/output_GAN_randomtasks_3.csv', 'output/random_dataset/final/landmark/output_static_woppie_3.csv', 'output/random_dataset/final/lastdrift/output_static_lastdrift_random_0.csv']
#datasets = ['output/bigloop_dataset/final/dualprompt/output_dualprompt_RecurrentTasks.csv_1.csv', 'output/bigloop_dataset/final/dynatraincdd/output_dynatraincdd_looploop_1.csv', 'output/bigloop_dataset/final/tfcl/prediction_results1.csv','output/bigloop_dataset/final/gan/output_GAN_recurrenttasks_4.csv','output/bigloop_dataset/final/landmark/output_static_looploop_1.csv', 'output/bigloop_dataset/final/lastdrift/output_lastdrift_loop_1.csv']
#datasets = ['output/infrequent/final/dualprompt/output_dualprompt_ImbalancedTasks.csv_3.csv', 'output/infrequent/final/dynatraincdd/output_dynatraincdd_imbalance_2.csv', 'output/infrequent/final/tfcl/prediction_results.csv', 'output/infrequent/final/gan/output_GAN_imbalancedtasks_4.csv' , 'output/infrequent/final/landmark/output_static_infrequent_3.csv', 'output/infrequent/final/lastdrift/output_staticlastdrift_infrequent_4.csv']
datasets = ['output/BPIC2015/final/dualprompt/output_dualprompt_BPIC15_recurrent_loop.csv_1.csv', 'output/BPIC2015/final/dynatraincdd/output_dynatraincdd_2015_0.csv', 'output/BPIC2015/final/tfcl/prediction_results3.csv', 'output/BPIC2015/final/gan/output_GAN_BPIC15_recurrent_loop_1.csv', 'output/BPIC2015/final/landmark/output_static_2015_2.csv', 'output/BPIC2015/final/lastdrift/output_static_lastdrift_2015_3.csv']
concept_drift = [2724, 5756, 9074, 12025, 15297, 18042, 20680, 23385, 26816, 30068, 33105, 36101, 38786, 41614, 44979, 47623, 50125, 52627, 55129, 57631, 60133, 63493, 66198, 69256, 72060, 75121, 77904]

differences = []
firsts = []
for i, dataset in enumerate(datasets):
    if 'random' in dataset:
        concept_drift = [2724, 5756, 9074, 12025, 15297, 18042, 20680, 23385, 26816, 30068, 33105, 36101, 38786, 41614, 44979, 47623, 50125, 52627, 55129, 57631, 60133, 63493, 66198, 69256, 72060, 75121, 77904]
    elif 'bigloop' in dataset:
        concept_drift = [2725, 5757, 9076, 12073, 15505, 18211, 21273, 24639, 27765, 31186, 33831, 36848, 40232, 42733, 45234, 47735, 50573, 53533, 56333, 59200, 61800, 64522, 67390, 70193, 73136, 75732, 78233, 80734, 83235, 85736, 88237, 90738, 93239, 95740, 98241]
    elif 'BPIC2015' in dataset:
        concept_drift = [2001, 4002, 6003, 8004, 10005, 12006, 14007, 16008, 18009, 20010, 22011, 24012, 26013, 28014, 30015]
    elif 'infrequent' in dataset:
        concept_drift = [2724, 5469, 8501, 11773, 14769, 17474, 19976, 22804, 25528, 28030]
    data = pd.read_csv(dataset)
    if titles[i] != 'DualPrompt' and titles[i] != 'GAN' and titles[i] != 'tfcl':
        results = []
        for row in data.iterrows():
            if row[1][1] == row[1][2]:
                results.append(1)
            else:
                results.append(0) 
        data = pd.DataFrame(calculate_percentage_of_ones(results))
        diff, first, alll = get_accs(data[0], concept_drift, dataset)
    elif titles[i] == 'GAN':
        data = pd.DataFrame(calculate_percentage_of_ones(list(data['0'])))
        diff, first, alll = get_accs(data[0], concept_drift, dataset)
    elif titles[i] == 'tfcl':
        results = []
        for row in data.iterrows():
            if row[1][0] == row[1][1]:
                results.append(1)
            else:
                results.append(0)
        data = pd.DataFrame(calculate_percentage_of_ones(results))
        diff, first, alll = get_accs(data[0], concept_drift, dataset)
    else:
        data = data.iloc[:,1]*100
        diff, first, alll = get_accs(data, concept_drift, dataset)
    differences.append(diff)
    firsts.append(first)

# Create a custom diverging colormap
colors = [(0, 'blue'), (0.5, 'white'), (1, 'red')]
custom_cmap = LinearSegmentedColormap.from_list("custom_diverging", colors)

Nr = 3
Nc = 2

# Create subplots for Nr rows and Nc columns
fig, axs = plt.subplots(Nr, Nc, figsize=(10, 12))
plt.subplots_adjust(hspace=0.4)
images = []

# Initialize vmin and vmax for color scaling
vmin = 100
vmax = 0
for i, data in enumerate(differences):
    row, col = divmod(i, Nc)
    img = axs[row, col].imshow(data, cmap=custom_cmap, aspect='auto')
    images.append(img)
    
    column_labels = [0,'1\n' + str(round(firsts[i][0],2)),'',  '2\n' + str(round(firsts[i][1],2)), ' ', '3\n'+str(round(firsts[i][2],2)),  '', '4\n'+str(round(firsts[i][3],2))]
    #column_labels = [0,'1\n' + str(round(firsts[i][0],2)), '2\n' + str(round(firsts[i][1],2)), '3\n'+str(round(firsts[i][2],2)), '4\n'+str(round(firsts[i][3],2)), '5\n'+str(round(firsts[i][4],2)), '6\n'+str(round(firsts[i][5],2)), '7\n'+str(round(firsts[i][6],2))]
    axs[row, col].set_xticklabels(column_labels, rotation=0)
    for k in range(data.shape[0]):
        for j in range(data.shape[1]):
            axs[row, col].text(j, k, f'{data[k, j]:.2f}', ha='center', va='center', color='black')
    axs[row, col].set_yticklabels([])
    axs[row, col].set_title(f'{titles[i]}')
    # Update vmin and vmax for color scaling
    vmin = min(vmin, np.min(data))
    vmax = max(vmax, np.max(data))

# Set the same color scale for all images
norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter = 0)
for img in images:
    img.set_norm(norm)

# Add colorbar
cbar_ax = fig.add_axes([0.15, -0.03, 0.7, 0.04])  # [left, bottom, width, height]
plt.colorbar(images[4], ax=cbar_ax, orientation='horizontal', fraction=1.5)
cbar_ax.yaxis.set_ticks([])
#cbar_ax.xaxis.set_ticks([])
cbar_ax.set_label('Accuracy Differences')
#plt.tight_layout()
#plt.colorbar().remove()
plt.show()

"""
methods = ['Dualprompt', 'No Prompt', 'G-Prompt', 'E-Prompt']
datasets = ['output/random_dataset/DualPrompt/output_dualprompt_2_wop_with_task_detection.csv', 'output/random_dataset/DualPrompt/output_noprompt_random_4.csv', 'output/random_dataset/DualPrompt/output_gprompt_random_4.csv', 'output/random_dataset/DualPrompt/output_eprompt_random_4.csv']
colors = ['skyblue', 'orange', 'green', 'red']

accuracy_values = []
for i in datasets:
    data = pd.read_csv(i)
    y = data.iloc[:,1]
    accuracy = (sum(y) / len(y))*100
    accuracy_values.append(accuracy)
 
# Create a bar plot
plt.figure(figsize=(8, 6))
bars = plt.bar(methods, accuracy_values, color=colors)
plt.ylim(0, 100)

# Add labels and title
plt.xlabel('Methods')
plt.ylabel('Accuracy (%)')
plt.title('Ablation Study: Accuracy of Different Methods')
for bar, accuracy in zip(bars, accuracy_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{accuracy:.1f}', ha='center', va='bottom')

# Show the plot
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()
"""