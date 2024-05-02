# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:30:59 2023

@author: tamar
"""

import torch.nn as nn

import pandas as pd
import matplotlib.pyplot as plt
"""
inp = pd.read_csv('local_datasets/OIR.csv')
print(len(inp))
def remove_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list

x = remove_duplicates(inp['case'])

def split_into_groups(input_set, group_size):
    input_list = list(input_set)
    groups = [input_list[i:i+group_size] for i in range(0, len(input_list), group_size)]
    return groups

groups = pd.DataFrame(split_into_groups(x, 500))
print(groups)
task1 = groups.loc[[0, 2, 4, 6, 8]]
task1 = [caseid for l in task1.values for caseid in l]
task2 = groups.loc[[1, 3, 5, 7, 9]]
task2 = [caseid for l in task2.values for caseid in l]
inp.loc[inp['case'].isin(task1),'task'] = 1 
inp.loc[inp['case'].isin(task2),'task'] = 2
inp.to_csv('local_datasets/OIR_tasks.csv')
df = pd.read_csv('local_datasets/OIR.csv')

# Convert 'completeTime' to datetime type
df['completeTime'] = pd.to_datetime(df['completeTime'], format='%H:%M.%S',errors='coerce')

# Sort the DataFrame based on 'case' and 'completeTime'
df_sorted = df.groupby('case', group_keys=False).apply(lambda x: x.sort_values('completeTime'))
"""
"""
def count_transitions_to_1(binary_list):
    count = []
    for i in range(1, len(binary_list)):
        if (binary_list[i - 1] != binary_list[i]):
            count.append(i)
    return count
get = pd.read_csv('local_datasets/loop_dataset.csv')
print(count_transitions_to_1(list(get['task'])))
"""
"""
output_dyna= pd.read_csv('output/dynatrainCDD/SDL_IOR_OTF_drift_dynamic.csv')
output_2 = pd.read_csv('output/tfcl/IOR_prediction_results.csv')
output_4= pd.read_csv('output/IOR parameters/hidden_size/output_hidden_100.csv')

results = []
for row in output_dyna.iterrows():
    if row[1][0] == row[1][1]:
        results.append(1)
    else:
        results.append(0)
        
results2 = []
for row in output_2.iterrows():
    if row[1][0] == row[1][1]:
        results2.append(1)
    else:
        results2.append(0)

def calculate_percentage_of_ones(data):
    window_size = 250
    result = []

    for i in range(len(data)):
        start_index = max(0, i - window_size + 1)
        window = data[start_index:i+1]
        percentage_of_ones = sum(window) / len(window) * 100
        result.append(percentage_of_ones)

    return result

x = calculate_percentage_of_ones(results)
y = range(500, len(x)+500)

x2 = calculate_percentage_of_ones(results2)
y2 = range(len(x2))
#x2 = output_2.iloc[:,0]
#y2 = output_2.iloc[:,1]
#x= output_dyna.iloc[:,0]
#y = output_dyna.iloc[:,1]
#x3= output_3.iloc[:,0]
#y3 = output_3.iloc[:,1]
x4= output_4.iloc[:,0]
y4 = output_4.iloc[:,1]
#x5 = output_5.iloc[:,0]
#y5 = output_5.iloc[:,1]
#concept_drift = [5224, 11042, 16247, 22178, 27322, 33187, 38349, 44269, 49451] #OIR
concept_drift =  [5224, 10469, 15674, 21002, 26146, 31429, 36591, 41853, 47035] #IOR
#concept_drift = [5818, 11042, 16860, 22065, 27996] #combi
# Plotting the line plot
plt.figure(figsize=(20,5))
plt.plot(y, x, linestyle='-', linewidth=1, color = 'orange', label = 'DynaTrainCDD')  
#plt.plot(x3, y3, linestyle='-', linewidth=1, color = 'purple', label='0.005')
plt.plot(y2, x2,linestyle='-', linewidth=1, color = 'green', label='tfcl')
plt.plot(x4, y4*100, linestyle='-', linewidth=1, color = 'red', label = 'DualPrompt')
#plt.plot(x5, y5, linestyle='-', linewidth=1, color = 'red', label='0.03')
for index in concept_drift:
    plt.axvline(x=index, color='red', linestyle='--', linewidth=1)  # Adjust color, linestyle, and linewidth as needed
c = [500, 5224, 10469, 15674, 21002, 26146, 31429, 36591, 41853, 47035, len(x4)+500] #combi
plt.axvspan(c[0], c[1], alpha=0.3, color = 'blue', label = 'Task 1')
plt.axvspan(c[2], c[3], alpha=0.3, color = 'blue')
plt.axvspan(c[4], c[5], alpha=0.3, color = 'blue')
plt.axvspan(c[6], c[7], alpha=0.3, color = 'blue')
plt.axvspan(c[8], c[9], alpha=0.3, color = 'blue')

plt.axvspan(c[1], c[2], alpha=0.3, color = 'green', label = 'Task 2')
plt.axvspan(c[3], c[4], alpha=0.3, color = 'green')
plt.axvspan(c[5], c[6], alpha=0.3, color = 'green')
plt.axvspan(c[7], c[8], alpha=0.3, color = 'green')
plt.axvspan(c[9], c[10], alpha=0.3, color = 'green')

plt.xticks(range(0, len(x), 10000))
plt.xlabel('Event')
plt.ylabel('Accuracy')
plt.title('Line Plot of the Data')
plt.grid(True)  # Add grid lines (optional)
plt.legend()
plt.show()

avg_acc = sum(x)/len(x)
print(f'DynaTrainCDD: {avg_acc}')
avg_acc = sum(x2)/len(x2)
print(f'tfcl: {avg_acc}')
#avg_acc = sum(y3)/len(y3)
#print(avg_acc)
avg_acc = sum(y4)/len(y4)
print(f'DualPrompt: {avg_acc*100}')
#avg_acc = sum(y5)/len(y5)
#print(avg_acc)
"""
"""
output_5= pd.read_csv('output/output_dataset_loop_dataset.csv')
output_1= pd.read_csv('output/tfcl/loop_dataset_prediction_results.csv')
output_2= pd.read_csv('output/dynatrainCDD/SDL_loop_dataset_OTF_drift_dynamic.csv')

results = []
for row in output_1.iterrows():
    if row[1][0] == row[1][1]:
        results.append(1)
    else:
        results.append(0)
        
results2 = []
for row in output_2.iterrows():
    if row[1][0] == row[1][1]:
        results2.append(1)
    else:
        results2.append(0)

def calculate_percentage_of_ones(data):
    window_size = 250
    result = []

    for i in range(len(data)):
        start_index = max(0, i - window_size + 1)
        window = data[start_index:i+1]
        percentage_of_ones = sum(window) / len(window) * 100
        result.append(percentage_of_ones)

    return result

x = calculate_percentage_of_ones(results)
y = range(500, len(x)+500)

x2 = calculate_percentage_of_ones(results2)
y2 = range(500, len(x2)+500)

x5 = output_5.iloc[:,0]
y5 = output_5.iloc[:,1]

concept_drift = [5224, 10756, 16574, 22070, 28001, 33206, 38767, 44632, 50257, 56177, 61321, 65513, 71396, 76892, 82823]# Plotting the line plot
plt.figure(figsize=(20,5))
plt.plot(y, x, linestyle='-', linewidth=1, color = 'black', label = 'tfcl')  
plt.plot(y2, x2,linestyle='-', linewidth=1, color = 'green', label='dynatrainCDD')
plt.plot(x5, y5*100, linestyle='-', linewidth=1, color = 'red', label='DualPrompt')
for index in concept_drift:
    plt.axvline(x=index, color='red', linestyle='--', linewidth=1)  # Adjust color, linestyle, and linewidth as needed
c = [500, 5224, 10756, 16574, 22070, 28001, 33206, 38767, 44632, 50257, 56177, 61321, 65513, 71396, 76892, 82823, len(x)+500] #combi
plt.axvspan(c[0], c[1], alpha=0.3, color = 'blue', label = 'Task 1')
plt.axvspan(c[5], c[6], alpha=0.3, color = 'blue')
plt.axvspan(c[10], c[11], alpha=0.3, color = 'blue')
plt.axvspan(c[15], c[16], alpha=0.3, color = 'blue')


plt.axvspan(c[1], c[2], alpha=0.3, color = 'purple', label = 'Task 2')
plt.axvspan(c[3], c[4], alpha=0.3, color = 'purple')
plt.axvspan(c[6], c[7], alpha=0.3, color = 'purple')
plt.axvspan(c[8], c[9], alpha=0.3, color = 'purple')
plt.axvspan(c[11], c[12], alpha=0.3, color = 'purple')
plt.axvspan(c[13], c[14], alpha=0.3, color = 'purple')

plt.axvspan(c[2], c[3], alpha=0.3, color = 'green', label = 'Task 3')
plt.axvspan(c[4], c[5], alpha=0.3, color = 'green')
plt.axvspan(c[7], c[8], alpha=0.3, color = 'green')
plt.axvspan(c[9], c[10], alpha=0.3, color = 'green')
plt.axvspan(c[12], c[13], alpha=0.3, color = 'green')
plt.axvspan(c[14], c[15], alpha=0.3, color = 'green')

plt.xticks(range(0, len(x), 5000))
plt.xlabel('Event')
plt.ylabel('Accuracy')
plt.title('Line Plot of the Data')
plt.grid(True)  # Add grid lines (optional)
plt.legend()
plt.show()

avg_acc = sum(x)/len(x)
print(f'tfcl: {avg_acc}')
avg_acc = sum(x2)/len(x2)
print(f'dynatrainCDD: {avg_acc}')
#avg_acc = sum(y3)/len(y3)
#print(avg_acc)
avg_acc = sum(y5)/len(y5)
print(f'DualPrompt: {avg_acc*100}')
#avg_acc = sum(y5)/len(y5)
#print(avg_acc)
"""
"""
output_5= pd.read_csv('output/all_datasets/output_buffer_size_100.csv')
output_1= pd.read_csv('output/tfcl/dataset_prediction_results.csv')
output_2= pd.read_csv('output/dynatrainCDD/SDL_dataset_OTF_drift_dynamic.csv')

results = []
for row in output_1.iterrows():
    if row[1][0] == row[1][1]:
        results.append(1)
    else:
        results.append(0)
        
results2 = []
for row in output_2.iterrows():
    if row[1][0] == row[1][1]:
        results2.append(1)
    else:
        results2.append(0)

def calculate_percentage_of_ones(data):
    window_size = 250
    result = []

    for i in range(len(data)):
        start_index = max(0, i - window_size + 1)
        window = data[start_index:i+1]
        percentage_of_ones = sum(window) / len(window) * 100
        result.append(percentage_of_ones)

    return result

x = calculate_percentage_of_ones(results)
y = range(500, len(x)+500)

x2 = calculate_percentage_of_ones(results2)
y2 = range(500, len(x2)+500)

x5 = output_5.iloc[:,0]
y5 = output_5.iloc[:,1]

concept_drift = [5224, 10756, 16575, 22027, 27233, 32766, 38012, 43151, 48357, 54289, 59786, 65324, 71190, 76687, 81873, 87018]
# Plotting the line plot
plt.figure(figsize=(20,5))
plt.plot(y, x, linestyle='-', linewidth=1, color = 'black', label = 'tfcl')  
plt.plot(y2, x2,linestyle='-', linewidth=1, color = 'green', label='dynatrainCDD')
plt.plot(x5, y5*100, linestyle='-', linewidth=1, color = 'red', label='DualPrompt')
for index in concept_drift:
    plt.axvline(x=index, color='red', linestyle='--', linewidth=1)  # Adjust color, linestyle, and linewidth as needed
c = [500, 5224, 10756, 16575, 22027, 27233, 32766, 38012, 43151, 48357, 54289, 59786, 65324, 71190, 76687, 81873, 87018, len(x)+500] #combi
plt.axvspan(c[0], c[1], alpha=0.3, color = 'blue', label = 'Task 1')
plt.axvspan(c[4], c[5], alpha=0.3, color = 'blue')
plt.axvspan(c[8], c[9], alpha=0.3, color = 'blue')
plt.axvspan(c[15], c[16], alpha=0.3, color = 'blue')

plt.axvspan(c[1], c[2], alpha=0.3, color = 'purple', label = 'Task 2')
plt.axvspan(c[13], c[14], alpha=0.3, color = 'purple')

plt.axvspan(c[2], c[3], alpha=0.3, color = 'green', label = 'Task 3')
plt.axvspan(c[9], c[10], alpha=0.3, color = 'green')
plt.axvspan(c[12], c[13], alpha=0.3, color = 'green')

plt.axvspan(c[3], c[4], alpha=0.3, color = 'orange', label = 'Task 4')
plt.axvspan(c[11], c[12], alpha=0.3, color = 'orange')

plt.axvspan(c[5], c[6], alpha=0.3, color = 'yellow', label = 'Task 5')
plt.axvspan(c[10], c[11], alpha=0.3, color = 'yellow')

plt.axvspan(c[6], c[7], alpha=0.3, color = 'red', label = 'Task 6')
plt.axvspan(c[16], c[17], alpha=0.3, color = 'red')

plt.axvspan(c[7], c[8], alpha=0.3, color = 'pink', label = 'Task 7')
plt.axvspan(c[14], c[15], alpha=0.3, color = 'pink')

plt.xticks(range(0, len(x), 5000))
plt.xlabel('Event')
plt.ylabel('Accuracy')
plt.title('Line Plot of the Data')
plt.grid(True)  # Add grid lines (optional)
plt.legend()
plt.show()

avg_acc = sum(x)/len(x)
print(f'tfcl: {avg_acc}')
avg_acc = sum(x2)/len(x2)
print(f'dynatrainCDD: {avg_acc}')
avg_acc = sum(y5)/len(y5)
print(f'DualPrompt: {avg_acc*100}')
"""
"""
output_dyna= pd.read_csv('output/dynatrainCDD/SDL_OIR_OTF_drift_dynamic.csv')
output_2 = pd.read_csv('output/tfcl/prediction_results.csv')
output_4= pd.read_csv('output/output_dataset_OIR_tasks.csv')

results = []
for row in output_dyna.iterrows():
    if row[1][0] == row[1][1]:
        results.append(1)
    else:
        results.append(0)
        
results2 = []
for row in output_2.iterrows():
    if row[1][0] == row[1][1]:
        results2.append(1)
    else:
        results2.append(0)

def calculate_percentage_of_ones(data):
    window_size = 250
    result = []

    for i in range(len(data)):
        start_index = max(0, i - window_size + 1)
        window = data[start_index:i+1]
        percentage_of_ones = sum(window) / len(window) * 100
        result.append(percentage_of_ones)

    return result

x = calculate_percentage_of_ones(results)
y = range(500, len(x)+500)

x2 = calculate_percentage_of_ones(results2)
y2 = range(len(x2))
#x2 = output_2.iloc[:,0]
#y2 = output_2.iloc[:,1]
#x= output_dyna.iloc[:,0]
#y = output_dyna.iloc[:,1]
#x3= output_3.iloc[:,0]
#y3 = output_3.iloc[:,1]
x4= output_4.iloc[:,0]
y4 = output_4.iloc[:,1]
#x5 = output_5.iloc[:,0]
#y5 = output_5.iloc[:,1]
concept_drift = [5224, 11042, 16247, 22178, 27322, 33187, 38349, 44269, 49451] #OIR
#concept_drift =  [5224, 10469, 15674, 21002, 26146, 31429, 36591, 41853, 47035] #IOR
#concept_drift = [5818, 11042, 16860, 22065, 27996] #combi
# Plotting the line plot
plt.figure(figsize=(20,5))
plt.plot(y, x, linestyle='-', linewidth=1, color = 'orange', label = 'DynaTrainCDD')  
#plt.plot(x3, y3, linestyle='-', linewidth=1, color = 'purple', label='0.005')
plt.plot(y2, x2,linestyle='-', linewidth=1, color = 'green', label='tfcl')
plt.plot(x4, y4*100, linestyle='-', linewidth=1, color = 'red', label = 'DualPrompt')
#plt.plot(x5, y5, linestyle='-', linewidth=1, color = 'red', label='0.03')
for index in concept_drift:
    plt.axvline(x=index, color='red', linestyle='--', linewidth=1)  # Adjust color, linestyle, and linewidth as needed
c = [500, 5224, 11042, 16247, 22178, 27322, 33187, 38349, 44269, 49451, len(x4)+500] #combi
plt.axvspan(c[0], c[1], alpha=0.3, color = 'blue', label = 'Task 1')
plt.axvspan(c[2], c[3], alpha=0.3, color = 'blue')
plt.axvspan(c[4], c[5], alpha=0.3, color = 'blue')
plt.axvspan(c[6], c[7], alpha=0.3, color = 'blue')
plt.axvspan(c[8], c[9], alpha=0.3, color = 'blue')

plt.axvspan(c[1], c[2], alpha=0.3, color = 'green', label = 'Task 2')
plt.axvspan(c[3], c[4], alpha=0.3, color = 'green')
plt.axvspan(c[5], c[6], alpha=0.3, color = 'green')
plt.axvspan(c[7], c[8], alpha=0.3, color = 'green')
plt.axvspan(c[9], c[10], alpha=0.3, color = 'green')

plt.xticks(range(0, len(x), 10000))
plt.xlabel('Event')
plt.ylabel('Accuracy')
plt.title('Line Plot of the Data')
plt.grid(True)  # Add grid lines (optional)
plt.legend()
plt.show()

avg_acc = sum(x)/len(x)
print(f'DynaTrainCDD: {avg_acc}')
avg_acc = sum(x2)/len(x2)
print(f'tfcl: {avg_acc}')
#avg_acc = sum(y3)/len(y3)
#print(avg_acc)
avg_acc = sum(y4)/len(y4)
print(f'DualPrompt: {avg_acc*100}')
#avg_acc = sum(y5)/len(y5)
#print(avg_acc)


df = pd.read_csv('local_datasets/OIR_tasks_gradual.csv')
# Create a dictionary to store tasks for each case
case_tasks = pd.DataFrame()

# Iterate through the list and populate the dictionary
for ind, item in df.iterrows():
    case = item['case']
    task = item['task']
    if ind == 0:
        case_tasks.loc[ind, 'task'] = task
        case_tasks.loc[ind, 'case'] = case
    elif case != df.loc[ind-1,'case']:
        case_tasks.loc[ind, 'task'] = task
        case_tasks.loc[ind, 'case'] = case

print(case_tasks)
tasks = case_tasks['task']

plt.figure(figsize=(20,5))
plt.scatter(range(350, 350+len(tasks[350:650])), tasks[350:650], marker = '.')
# Set labels for x and y axes
plt.xticks(range(350, 650, 50))
plt.yticks([1, 2])
plt.xlabel('Case Index')
plt.ylabel('Task')

# Show the plot
plt.show()
"""
"""
output_1 = pd.read_csv('output/combined parameters/output_buffer_size_50.csv')
output_2 = pd.read_csv('output/combined parameters/output_buf_25.csv')
output_3 = pd.read_csv('output/combined parameters/output_buffer_size_100.csv')
output_4 = pd.read_csv('output/combined parameters/output_buffer_size_200.csv')
output_5= pd.read_csv('output/combined parameters/output_buffer_size_250.csv')

x2 = output_2.iloc[:,0]
y2 = output_2.iloc[:,1]
x= output_1.iloc[:,0]
y = output_1.iloc[:,1]
x3= output_3.iloc[:,0]
y3 = output_3.iloc[:,1]
x4= output_4.iloc[:,0]
y4 = output_4.iloc[:,1]
x5 = output_5.iloc[:,0]
y5 = output_5.iloc[:,1]

concept_drift =  [5818, 11042, 16860, 22065, 27996] #combi
# Plotting the line plot
plt.figure(figsize=(20,5))
plt.plot(x, y*100, linestyle='-', linewidth=1, color = 'orange', label = '50')  
plt.plot(x3, y3*100, linestyle='-', linewidth=1, color = 'purple', label='100')
plt.plot(x2, y2*100,linestyle='-', linewidth=1, color = 'green', label='25')
plt.plot(x4, y4*100, linestyle='-', linewidth=1, color = 'blue', label = '200')
plt.plot(x5, y5*100, linestyle='-', linewidth=1, color = 'red', label='250')
for index in concept_drift:
    plt.axvline(x=index, color='red', linestyle='--', linewidth=1)  # Adjust color, linestyle, and linewidth as needed
c = [500, 5818, 11042, 16860, 22065, 27996, len(x)+500] #combi
plt.axvspan(c[0], c[1], alpha=0.3, color = 'blue', label = 'Task 1')
plt.axvspan(c[5], c[6], alpha=0.3, color = 'blue')

plt.axvspan(c[2], c[3], alpha=0.3, color = 'purple', label = 'Task 3')
plt.axvspan(c[4], c[5], alpha=0.3, color = 'purple')

plt.axvspan(c[1], c[2], alpha=0.3, color = 'green', label = 'Task 2')
plt.axvspan(c[3], c[4], alpha=0.3, color = 'green')

plt.xticks(range(0, len(x), 5000))
plt.xlabel('Event')
plt.ylabel('Accuracy')
plt.title('Line Plot of the Data')
plt.grid(True)  # Add grid lines (optional)
plt.legend()
plt.show()

avg_acc = sum(y)/len(y)
print(avg_acc)
avg_acc = sum(y2)/len(y2)
print(avg_acc)
avg_acc = sum(y3)/len(y3)
print(avg_acc)
avg_acc = sum(y4)/len(y4)
print(avg_acc)
avg_acc = sum(y5)/len(y5)
print(avg_acc)
"""

"""
output_1 = pd.read_csv('output/bigloop_dataset/output_with_forgetting.csv')
output_2 = pd.read_csv('output/bigloop_dataset/output_with_forgetting_loop_noE.csv')
output_3 = pd.read_csv('output/bigloop_dataset/output_with_forgetting_loop_noG.csv')
output_4 = pd.read_csv('output/bigloop_dataset/output_with_forgetting_loop_none.csv')
output_5 = pd.read_csv('output/bigloop_dataset/SDL_looooop_OTF_drift_dynamic.csv')
output_6 = pd.read_csv('output/bigloop_dataset/prediction_results.csv')
"""
"""
results = []
for row in output_5.iterrows():
    if row[1][0] == row[1][1]:
        results.append(1)
    else:
        results.append(0)
        
results2 = []
for row in output_6.iterrows():
    if row[1][0] == row[1][1]:
        results2.append(1)
    else:
        results2.append(0)
        
def calculate_percentage_of_ones(data):
    window_size = 250
    result = []

    for i in range(len(data)):
        start_index = max(0, i - window_size + 1)
        window = data[start_index:i+1]
        percentage_of_ones = sum(window) / len(window) * 100
        result.append(percentage_of_ones)

    return result

x5 = calculate_percentage_of_ones(results)
y5 = range(500, len(x5)+500)
x6 = calculate_percentage_of_ones(results2)
y6 = range(500, len(x6)+500)
"""
"""
x= output_1.iloc[:,0]
y = output_1.iloc[:,1]
x2 = output_2.iloc[:,0]
y2 = output_2.iloc[:,1]
x3= output_3.iloc[:,0]
y3 = output_3.iloc[:,1]
x4= output_4.iloc[:,0]
y4 = output_4.iloc[:,1]

concept_drift =  [2725, 5757, 9075, 12071, 15502, 18207, 21268, 24633, 27758, 31178, 33822, 36838, 40221, 42722, 45223, 47725, 50563, 53523, 56323, 59190, 61790, 64512, 67380, 70182, 73124, 75720, 78222, 80724, 83226, 85728, 88230, 90732, 93234, 95736, 98238] #combi
# Plotting the line plot
plt.figure(figsize=(20,5))
plt.plot(x, y*100, linestyle='-', linewidth=1, color = 'green', label = 'Both')  
plt.plot(x2, y2*100, linestyle='-', linewidth=1, color = 'red', label='G-Prompt')
plt.plot(x3, y3*100,linestyle='-', linewidth=1, color = 'purple', label='E-prompt')
plt.plot(x4, y4*100, linestyle='-', linewidth=1, color = 'blue', label = 'None')  
#plt.plot(y5, x5, linestyle='-', linewidth=1, color = 'orange', label='DynaTrainCDD')
#plt.plot(y6, x6, linestyle='-', linewidth=1, color = 'yellow', label='tfcl')

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

plt.xticks(range(0, len(x), 5000))
plt.xlabel('Event')
plt.ylabel('Accuracy')
plt.title('Line Plot of the Data')
plt.grid(True)  # Add grid lines (optional)
plt.legend()
plt.show()

avg_acc = sum(y)/len(y)
print(avg_acc*100)
avg_acc = sum(y2)/len(y2)
print(avg_acc*100)
avg_acc = sum(y3)/len(y3)
print(avg_acc*100)
avg_acc = sum(y4)/len(y4)
print(avg_acc*100)
avg_acc = sum(x5)/len(x5)
print(avg_acc)
avg_acc = sum(x6)/len(x6)
print(avg_acc)

output_1 = pd.read_csv('output/dataset paper/output_dualprompt_cp.csv')
output_2 = pd.read_csv('output/dataset paper/output_noprompt_cp.csv')
output_3 = pd.read_csv('output/dataset paper/output_gprompt_cp.csv')
output_4 = pd.read_csv('output/dataset paper/output_eprompt_cp.csv')
color_input = pd.read_csv('local_datasets/cp_recurring_tasks.csv')
x= output_1.iloc[:,0]
y = output_1.iloc[:,1]
x2= output_2.iloc[:,0]
y2 = output_2.iloc[:,1]
x3= output_3.iloc[:,0]
y3 = output_3.iloc[:,1]
x4= output_4.iloc[:,0]
y4 = output_4.iloc[:,1]
color_values = color_input['task'].values

plt.figure(figsize=(20,5))
plt.plot(x, y*100, linestyle='-', linewidth=1, color = 'red', label = 'DualPrompt')  
plt.plot(x2, y2*100, linestyle='-', linewidth=1, color = 'yellow', label = 'No updating')  
plt.plot(x3, y3*100, linestyle='-', linewidth=1, color = 'orange', label = 'G-Prompt')  
plt.plot(x4, y4*100, linestyle='-', linewidth=1, color = 'white', label = 'E-Prompt')

for i in range(len(x)):
    if color_values[i] == 1:
        plt.axvspan(x[i], x[i]+(x[1]-x[0]), ymin=0, ymax=100, alpha=0.8, color='green')  # Adjust alpha for transparency
    elif color_values[i] == 2:
        plt.axvspan(x[i], x[i]+(x[1]-x[0]), ymin=0, ymax=100, alpha=0.8, color='blue')  # Adjust alpha for transparency
    
plt.xticks(range(0, len(x), 5000))
plt.xlabel('Event')
plt.ylabel('Accuracy')
plt.title('Line Plot of the Data')
plt.grid(True)  # Add grid lines (optional)
plt.legend()
plt.show()

avg_acc = sum(y)/len(y)
print(avg_acc*100)
avg_acc = sum(y2)/len(y2)
print(avg_acc*100)

output_1 = pd.read_csv('output/dataset paper/output_dualprompt_lp.csv')
output_2 = pd.read_csv('output/dataset paper/output_noprompt_lp.csv')
output_3 = pd.read_csv('output/dataset paper/output_gprompt_lp.csv')
output_4 = pd.read_csv('output/dataset paper/output_eprompt_lp.csv')
color_input = pd.read_csv('local_datasets/lp_recurring_tasks.csv')
x= output_1.iloc[:,0]
y = output_1.iloc[:,1]
x2= output_2.iloc[:,0]
y2 = output_2.iloc[:,1]
x3= output_3.iloc[:,0]
y3 = output_3.iloc[:,1]
x4= output_4.iloc[:,0]
y4 = output_4.iloc[:,1]
color_values = color_input['task'].values

plt.figure(figsize=(20,5))
plt.plot(x, y*100, linestyle='-', linewidth=1, color = 'red', label = 'DualPrompt')  
plt.plot(x2, y2*100, linestyle='-', linewidth=1, color = 'yellow', label = 'No updating')  
plt.plot(x3, y3*100, linestyle='-', linewidth=1, color = 'orange', label = 'G-Prompt')  
plt.plot(x4, y4*100, linestyle='-', linewidth=1, color = 'white', label = 'E-Prompt')

for i in range(len(x)):
    if color_values[i] == 1:
        plt.axvspan(x[i], x[i]+(x[1]-x[0]), ymin=0, ymax=100, alpha=0.8, color='green')  # Adjust alpha for transparency
    elif color_values[i] == 2:
        plt.axvspan(x[i], x[i]+(x[1]-x[0]), ymin=0, ymax=100, alpha=0.8, color='blue')  # Adjust alpha for transparency
   
plt.xticks(range(0, len(x), 5000))
plt.xlabel('Event')
plt.ylabel('Accuracy')
plt.title('Line Plot of the Data')
plt.grid(True)  # Add grid lines (optional)
plt.legend()
plt.show()

avg_acc = sum(y)/len(y)
print(avg_acc*100)
avg_acc = sum(y2)/len(y2)
print(avg_acc*100)

output_1 = pd.read_csv('output/dataset paper/output_dualprompt_OIR.csv')
output_2 = pd.read_csv('output/dataset paper/output_noprompt_OIR.csv')
output_3 = pd.read_csv('output/dataset paper/output_gprompt_OIR.csv')
output_4 = pd.read_csv('output/dataset paper/output_eprompt_OIR.csv')

color_input = pd.read_csv('local_datasets/OIR_recurring_tasks.csv')
x= output_1.iloc[:,0]
y = output_1.iloc[:,1]
x2= output_2.iloc[:,0]
y2 = output_2.iloc[:,1]
x3= output_3.iloc[:,0]
y3 = output_3.iloc[:,1]
x4= output_4.iloc[:,0]
y4 = output_4.iloc[:,1]
color_values = color_input['task'].values

plt.figure(figsize=(20,5))
plt.plot(x, y*100, linestyle='-', linewidth=1, color = 'red', label = 'DualPrompt')  
plt.plot(x2, y2*100, linestyle='-', linewidth=1, color = 'yellow', label = 'No updating')  
plt.plot(x3, y3*100, linestyle='-', linewidth=1, color = 'orange', label = 'G-Prompt')  
plt.plot(x4, y4*100, linestyle='-', linewidth=1, color = 'white', label = 'E-Prompt') 

for i in range(len(x)):
    if color_values[i] == 1:
        plt.axvspan(x[i], x[i]+(x[1]-x[0]), ymin=0, ymax=100, alpha=0.8, color='green')  # Adjust alpha for transparency
    elif color_values[i] == 2:
        plt.axvspan(x[i], x[i]+(x[1]-x[0]), ymin=0, ymax=100, alpha=0.8, color='blue')  # Adjust alpha for transparency
   
plt.xticks(range(0, len(x), 5000))
plt.xlabel('Event')
plt.ylabel('Accuracy')
plt.title('Line Plot of the Data')
plt.grid(True)  # Add grid lines (optional)
plt.legend()
plt.show()

avg_acc = sum(y)/len(y)
print(avg_acc*100)
avg_acc = sum(y2)/len(y2)
print(avg_acc*100)
"""
