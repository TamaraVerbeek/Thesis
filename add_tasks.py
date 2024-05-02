# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:16:40 2023

@author: tamar
"""
import pandas as pd
"""
x = pd.read_csv('./local_datasets/BPIC2020/PrepaidTravelCosts.csv', low_memory=False,  encoding='latin-1')
# Convert 'completeTime' column to datetime format
x['completeTime'] = pd.to_datetime(x['completeTime'], format='%Y/%m/%d %H:%M:%S.%f')

# Sort the DataFrame based on 'completeTime' column
df_sorted = x.sort_values(by='completeTime')

# Save the sorted DataFrame to a new CSV file
df_sorted.to_csv('local_datasets/BPIC2020/PrepaidTravelCosts.csv', index=False)

cases = x['case']
x['task'] = x['case'].apply(lambda x: 2 if 330 <= x < 660 else 1)
x.to_csv('./local_datasets/cp_recurring_tasks.csv', index=False)
"""
"""
wop = pd.read_csv('./local_datasets/looploop_dataset.csv', low_memory=False, encoding='latin-1')
bpi15_recur = pd.read_csv('./local_datasets/BPIC15_recurrent_loop.csv', low_memory=False, encoding='latin-1')
infrequent = pd.read_csv('./local_datasets/self made datasets/infrequent_tasks.csv', low_memory=False, encoding='latin-1')

IRO_2 = pd.read_csv('./local_datasets/IRO_2.5k_tasks.csv', low_memory=False,  encoding='latin-1')
IRO = pd.read_csv('./local_datasets/IRO_tasks.csv', low_memory=False,  encoding='latin-1')

OIR_2 = pd.read_csv('./local_datasets/OIR_2.5k_tasks.csv', low_memory=False,  encoding='latin-1')
OIR = pd.read_csv('./local_datasets/OIR_tasks.csv', low_memory=False,  encoding='latin-1')

ORI = pd.read_csv('./local_datasets/ORI_tasks.csv', low_memory=False,  encoding='latin-1')
IOR = pd.read_csv('./local_datasets/IOR_tasks.csv', low_memory=False,  encoding='latin-1')
RIO = pd.read_csv('./local_datasets/RIO_tasks.csv', low_memory=False,  encoding='latin-1')
ROI = pd.read_csv('./local_datasets/ROI_tasks.csv', low_memory=False,  encoding='latin-1')

def count_transitions_to_1(binary_list):
    count = []
    for i in range(1, len(binary_list)):
        if (binary_list[i - 1] != binary_list[i]):
            count.append(i)
    return count
wop = count_transitions_to_1(list(infrequent['task']))
print(wop)
"""

x1 = pd.read_csv('./local_datasets/BPIC2015/BPIC15_1.csv', low_memory=False,  encoding='latin-1')
x1.sort_values(by='case')
x2 = pd.read_csv('./local_datasets/BPIC2015/BPIC15_2.csv', low_memory=False,  encoding='latin-1')
x2.sort_values(by='case')

x3 = pd.read_csv('./local_datasets/BPIC2015/BPIC15_3.csv', low_memory=False,  encoding='latin-1')
x3.sort_values(by='case')

x4 = pd.read_csv('./local_datasets/BPIC2015/BPIC15_4.csv', low_memory=False,  encoding='latin-1')
x4.sort_values(by='case')

x5 = pd.read_csv('./local_datasets/BPIC2015/BPIC15_5.csv', low_memory=False,  encoding='latin-1')
x5.sort_values(by='case')

x1.loc[0:2000, 'task'] = 1
new_dataset = x1.loc[0:4000] #1
x2.loc[0:2000, 'task'] = 2
new_dataset = pd.concat([new_dataset, x2.loc[0:4000]], ignore_index= True) 
x3.loc[0:2000, 'task'] = 3
new_dataset = pd.concat([new_dataset, x3.loc[0:4000]], ignore_index= True)
x4.loc[0:2000, 'task'] = 4
new_dataset = pd.concat([new_dataset, x4.loc[0:4000]], ignore_index= True)

x1.loc[4000:6000, 'task'] = 1
new_dataset = pd.concat([new_dataset, x1.loc[4000:6000]], ignore_index= True) 
x2.loc[4000:6000, 'task'] = 2
new_dataset = pd.concat([new_dataset, x2.loc[4000:6000]], ignore_index= True) 
x3.loc[4000:6000, 'task'] = 3
new_dataset = pd.concat([new_dataset, x3.loc[4000:6000]], ignore_index= True)
x4.loc[4000:6000, 'task'] = 4
new_dataset = pd.concat([new_dataset, x4.loc[4000:6000]], ignore_index= True)

x1.loc[6000:8000, 'task'] = 1
new_dataset = pd.concat([new_dataset, x1.loc[6000:8000]], ignore_index= True) 
x2.loc[6000:8000, 'task'] = 2
new_dataset = pd.concat([new_dataset, x2.loc[6000:8000]], ignore_index= True) 
x3.loc[6000:8000, 'task'] = 3
new_dataset = pd.concat([new_dataset, x3.loc[6000:8000]], ignore_index= True)
x4.loc[6000:8000, 'task'] = 4
new_dataset = pd.concat([new_dataset, x4.loc[6000:8000]], ignore_index= True)

x1.loc[2000:4000, 'task'] = 1
new_dataset = pd.concat([new_dataset, x1.loc[8000:10000]], ignore_index= True) 
x2.loc[2000:4000, 'task'] = 2
new_dataset = pd.concat([new_dataset, x2.loc[8000:10000]], ignore_index= True) 
x3.loc[2000:4000, 'task'] = 3
new_dataset = pd.concat([new_dataset, x3.loc[8000:10000]], ignore_index= True)
x4.loc[2000:4000, 'task'] = 4
new_dataset = pd.concat([new_dataset, x4.loc[8000:10000]], ignore_index= True)


new_dataset.to_csv('./local_datasets/BPIC15_recurrent_loop_2.csv', index=False) 
"""
IRO_concepts = count_transitions_to_1(list(IRO['task']))
#OIR_concepts = count_transitions_to_1(list(OIR['task']))
ORI_concepts = count_transitions_to_1(list(ORI['task']))
IOR_concepts = count_transitions_to_1(list(IOR['task']))
#RIO_concepts = count_transitions_to_1(list(RIO['task']))
#ROI_concepts = count_transitions_to_1(list(ROI['task']))

#IRO_2_concepts = count_transitions_to_1(list(IRO_2['task']))
#OIR_2_concepts = count_transitions_to_1(list(OIR_2['task']))

#4 x 1, 2 x 2, 3 x 3, 1 x 4

new_dataset = IOR.loc[0:IOR_concepts[0]-2501] #1
IOR.loc[IOR_concepts[0]:IOR_concepts[1]-2501, 'task'] = 3
new_dataset = pd.concat([new_dataset, IOR.loc[IOR_concepts[0]:IOR_concepts[1]-2501]], ignore_index= True) #3
IRO.loc[IRO_concepts[0] :IRO_concepts[1]-2501,'task'] = 2
new_dataset = pd.concat([new_dataset, IRO.loc[IRO_concepts[0] :IRO_concepts[1]-2501]], ignore_index= True)
ORI.loc[ORI_concepts[0] :ORI_concepts[1]-2501, 'task'] = 4
new_dataset = pd.concat([new_dataset, ORI.loc[ORI_concepts[0] :ORI_concepts[1]-2501]], ignore_index= True)
IRO.loc[IRO_concepts[2] :IRO_concepts[3]-2501, 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO.loc[IRO_concepts[2] :IRO_concepts[3]-2501]], ignore_index= True)
IOR.loc[IOR_concepts[0]-2501:IOR_concepts[0], 'task'] = 1
new_dataset = pd.concat([new_dataset, IOR.loc[IOR_concepts[1] :IOR_concepts[2]-2501]], ignore_index= True)
IRO.loc[IRO_concepts[3]-2501:IRO_concepts[3] , 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO.loc[IRO_concepts[3]-2501:IRO_concepts[3] ]], ignore_index= True)
IOR.loc[IOR_concepts[2] :IOR_concepts[3]-2501, 'task'] = 3
new_dataset = pd.concat([new_dataset, IOR.loc[IOR_concepts[2] :IOR_concepts[3]-2501]], ignore_index= True)
new_dataset = pd.concat([new_dataset, IRO.loc[0:IRO_concepts[0]-2501]], ignore_index= True) #1
IOR.loc[IOR_concepts[3]-2501:IOR_concepts[3] , 'task'] = 3
new_dataset = pd.concat([new_dataset, IOR.loc[IOR_concepts[3]-2501:IOR_concepts[3] ]], ignore_index= True)
new_dataset = pd.concat([new_dataset, ORI.loc[0:ORI_concepts[0]-2501]], ignore_index= True) #1
new_dataset.to_csv('./local_datasets/infrequent_tasks.csv', index=False) 
"""

"""
new_dataset = IRO.loc[0:IRO_concepts[0]-2501] #1 
new_dataset = pd.concat([new_dataset, IRO.loc[IRO_concepts[0]:IRO_concepts[1]-2501]], ignore_index= True) #2
OIR.loc[OIR_concepts[0] :OIR_concepts[1]-2501,'task'] = 3
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[0] :OIR_concepts[1]-2501]], ignore_index= True)
RIO.loc[RIO_concepts[0] :RIO_concepts[1]-2501, 'task'] = 4
new_dataset = pd.concat([new_dataset, RIO.loc[RIO_concepts[0] :RIO_concepts[1]-2501]], ignore_index= True)
ORI.loc[ORI_concepts[0] :ORI_concepts[1]-2501, 'task'] = 5
new_dataset = pd.concat([new_dataset, ORI.loc[ORI_concepts[0] :ORI_concepts[1]-2501]], ignore_index= True)
IOR.loc[IOR_concepts[0] :IOR_concepts[1]-2501, 'task'] = 6
new_dataset = pd.concat([new_dataset, IOR.loc[IOR_concepts[0] :IOR_concepts[1]-2501]], ignore_index= True)
ROI.loc[ROI_concepts[0] :ROI_concepts[1]-2501, 'task'] = 7
new_dataset = pd.concat([new_dataset, ROI.loc[ROI_concepts[0] :ROI_concepts[1]-2501]], ignore_index= True)

ROI.loc[ROI_concepts[1] :ROI_concepts[2]-2501, 'task'] = 1
new_dataset = pd.concat([new_dataset, ROI.loc[ROI_concepts[1] :ROI_concepts[2]-2501]], ignore_index= True)
OIR.loc[OIR_concepts[2] :OIR_concepts[3]-2501, 'task'] = 3
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[2] :OIR_concepts[3]-2501]], ignore_index= True)
ORI.loc[ORI_concepts[2] :ORI_concepts[3]-2501, 'task'] = 5
new_dataset = pd.concat([new_dataset, ORI.loc[ORI_concepts[2] :ORI_concepts[3]-2501]], ignore_index= True)
RIO.loc[RIO_concepts[2] :RIO_concepts[3]-2501, 'task'] = 4
new_dataset = pd.concat([new_dataset, RIO.loc[RIO_concepts[2] :RIO_concepts[3]-2501]], ignore_index= True)
IRO.loc[IRO_concepts[2] :IRO_concepts[3]-2501, 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO.loc[IRO_concepts[2] :IRO_concepts[3]-2501]], ignore_index= True)
ROI.loc[ROI_concepts[2] :ROI_concepts[3]-2501, 'task'] = 7
new_dataset = pd.concat([new_dataset, ROI.loc[ROI_concepts[2] :ROI_concepts[3]-2501]], ignore_index= True)
IOR.loc[IOR_concepts[2] :IOR_concepts[3]-2501, 'task'] = 6
new_dataset = pd.concat([new_dataset, IOR.loc[IOR_concepts[2] :IOR_concepts[3]-2501]], ignore_index= True)

OIR.loc[OIR_concepts[4] :OIR_concepts[5]-2501, 'task'] = 3
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[4] :OIR_concepts[5]-2501]], ignore_index= True)
ROI.loc[ROI_concepts[3] :ROI_concepts[4]-2501, 'task'] = 1
new_dataset = pd.concat([new_dataset, ROI.loc[ROI_concepts[3] :ROI_concepts[4]-2501]], ignore_index= True)
IRO.loc[IRO_concepts[3]-2501:IRO_concepts[3] , 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO.loc[IRO_concepts[3]-2501:IRO_concepts[3] ]], ignore_index= True)
ROI.loc[ROI_concepts[3]-2501:ROI_concepts[3] , 'task'] = 7
new_dataset = pd.concat([new_dataset, ROI.loc[ROI_concepts[3]-2501:ROI_concepts[3] ]], ignore_index= True)
ORI.loc[ORI_concepts[3]-2501:ORI_concepts[3] , 'task'] = 5
new_dataset = pd.concat([new_dataset, ORI.loc[ORI_concepts[3]-2501:ORI_concepts[3] ]], ignore_index= True)
IOR.loc[IOR_concepts[3]-2501:IOR_concepts[3] , 'task'] = 6
new_dataset = pd.concat([new_dataset, IOR.loc[IOR_concepts[3]-2501:IOR_concepts[3] ]], ignore_index= True)
RIO.loc[RIO_concepts[3]-2501:RIO_concepts[3] , 'task'] = 4
new_dataset = pd.concat([new_dataset, RIO.loc[RIO_concepts[3]-2501:RIO_concepts[3] ]], ignore_index= True)


ORI.loc[ORI_concepts[4] :ORI_concepts[5]-2501, 'task'] = 5
new_dataset = pd.concat([new_dataset, ORI.loc[ORI_concepts[4] :ORI_concepts[5]-2501]], ignore_index= True)
OIR.loc[OIR_concepts[1] :OIR_concepts[2]-2501, 'task'] = 1
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[1] :OIR_concepts[2]-2501]], ignore_index= True)
RIO.loc[RIO_concepts[4] :RIO_concepts[5]-2501, 'task'] = 4
new_dataset = pd.concat([new_dataset, RIO.loc[RIO_concepts[4] :RIO_concepts[5]-2501]], ignore_index= True)
ROI.loc[ROI_concepts[4] :ROI_concepts[5]-2501, 'task'] = 7
new_dataset = pd.concat([new_dataset, ROI.loc[ROI_concepts[4] :ROI_concepts[5]-2501]], ignore_index= True)
IRO.loc[IRO_concepts[4] :IRO_concepts[5]-2501, 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO.loc[IRO_concepts[4] :IRO_concepts[5]-2501]], ignore_index= True)
IOR.loc[IOR_concepts[4] :IOR_concepts[5]-2501, 'task'] = 6
new_dataset = pd.concat([new_dataset, IOR.loc[IOR_concepts[4] :IOR_concepts[5]-2501]], ignore_index= True)
OIR.loc[OIR_concepts[5]-2501:OIR_concepts[5] , 'task'] = 3
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[5]-2501:OIR_concepts[5] ]], ignore_index= True)
new_dataset.to_csv('./local_datasets/newwoppie.csv', index=False) 
"""
"""
new_dataset = IRO.loc[0:IRO_concepts[0]-2500] #1 
new_dataset = pd.concat([new_dataset, IRO.loc[IRO_concepts[0]:IRO_concepts[1]-2501]], ignore_index= True) #2
OIR.loc[OIR_concepts[0] :OIR_concepts[1]-2501,'task'] = 3
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[0] :OIR_concepts[1]-2501]], ignore_index= True)
IRO.loc[IRO_concepts[2] :IRO_concepts[3]-2501, 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO.loc[IRO_concepts[2] :IRO_concepts[3]-2501]], ignore_index= True)
OIR.loc[OIR_concepts[2] :OIR_concepts[3]-2501, 'task'] = 3
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[2] :OIR_concepts[3]-2501]], ignore_index= True)
OIR.loc[OIR_concepts[1] :OIR_concepts[2]-2501, 'task'] = 1
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[1] :OIR_concepts[2]-2501]], ignore_index= True)

IRO.loc[IRO_concepts[4] :IRO_concepts[5]-2501, 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO.loc[IRO_concepts[4] :IRO_concepts[5]-2501]], ignore_index= True)
OIR.loc[OIR_concepts[4] :OIR_concepts[5]-2501,'task'] = 3
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[4] :OIR_concepts[5]-2501]], ignore_index= True)
IRO.loc[IRO_concepts[6] :IRO_concepts[7]-2501, 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO.loc[IRO_concepts[6] :IRO_concepts[7]-2501]], ignore_index= True)
OIR.loc[OIR_concepts[6] :OIR_concepts[7]-2501, 'task'] = 3
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[6] :OIR_concepts[7]-2501]], ignore_index= True)
OIR.loc[OIR_concepts[3] :OIR_concepts[4]-2501, 'task'] = 1
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[3] :OIR_concepts[4]-2501]], ignore_index= True)

IRO.loc[IRO_concepts[8] :len(IRO)-2501, 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO.loc[IRO_concepts[8] :len(IRO)-2501]], ignore_index= True)
OIR.loc[OIR_concepts[8] :len(OIR)-2501,'task'] = 3
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[8] :len(OIR)-2501]], ignore_index= True)
IRO.loc[len(IRO)-2501:len(IRO) , 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO.loc[len(IRO)-2501:len(IRO) ]], ignore_index= True)
OIR.loc[len(OIR)-2501:len(OIR) ,'task'] = 3
new_dataset = pd.concat([new_dataset, OIR.loc[len(OIR)-2501:len(OIR) ]], ignore_index= True)
OIR.loc[OIR_concepts[6]-2501:OIR_concepts[6] , 'task'] = 1
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[6]-2501:OIR_concepts[6] ]], ignore_index= True)

IRO_2.loc[IRO_2_concepts[2] :IRO_2_concepts[3] , 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO_2.loc[IRO_2_concepts[2] :IRO_2_concepts[3] ]], ignore_index= True)
OIR_2.loc[OIR_2_concepts[2] :OIR_2_concepts[3] , 'task'] = 3
new_dataset = pd.concat([new_dataset, OIR_2.loc[OIR_2_concepts[2] :OIR_2_concepts[3] ]], ignore_index= True)
IRO_2.loc[IRO_2_concepts[4] :IRO_2_concepts[5] , 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO_2.loc[IRO_2_concepts[4] :IRO_2_concepts[5] ]], ignore_index= True)
OIR_2.loc[OIR_2_concepts[4] :OIR_2_concepts[5] ,'task'] = 3
new_dataset = pd.concat([new_dataset, OIR_2.loc[OIR_2_concepts[4] :OIR_2_concepts[5] ]], ignore_index= True)
OIR_2.loc[OIR_2_concepts[1] :OIR_2_concepts[2] , 'task'] = 1
new_dataset = pd.concat([new_dataset, OIR_2.loc[OIR_2_concepts[1] :OIR_2_concepts[2] ]], ignore_index= True)

IRO_2.loc[IRO_2_concepts[6] :IRO_2_concepts[7] , 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO_2.loc[IRO_2_concepts[6] :IRO_2_concepts[7] ]], ignore_index= True)
OIR_2.loc[OIR_2_concepts[6] :OIR_2_concepts[7] , 'task'] = 3
new_dataset = pd.concat([new_dataset, OIR_2.loc[OIR_2_concepts[6] :OIR_2_concepts[7] ]], ignore_index= True)
IRO_2.loc[IRO_2_concepts[8] :len(IRO_2) , 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO_2.loc[IRO_2_concepts[8] :len(IRO_2) ]], ignore_index= True)
OIR_2.loc[OIR_2_concepts[8] :len(OIR_2) ,'task'] = 3
new_dataset = pd.concat([new_dataset, OIR_2.loc[OIR_2_concepts[8] :len(OIR_2) ]], ignore_index= True)
OIR_2.loc[OIR_2_concepts[3] :OIR_2_concepts[4] , 'task'] = 1
new_dataset = pd.concat([new_dataset, OIR_2.loc[OIR_2_concepts[3] :OIR_2_concepts[4] ]], ignore_index= True)

IRO.loc[IRO_concepts[3]-2501:IRO_concepts[3] , 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO.loc[IRO_concepts[3]-2501:IRO_concepts[3] ]], ignore_index= True)
OIR.loc[OIR_concepts[1]-2501:OIR_concepts[1] ,'task'] = 3
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[1]-2501:OIR_concepts[1] ]], ignore_index= True)
IRO.loc[IRO_concepts[1]-2501:IRO_concepts[1] , 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO.loc[IRO_concepts[1]-2501:IRO_concepts[1] ]], ignore_index= True)
OIR.loc[OIR_concepts[3]-2501:OIR_concepts[3] , 'task'] = 3
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[3]-2501:OIR_concepts[3] ]], ignore_index= True)
OIR.loc[OIR_concepts[2]-2501:OIR_concepts[2] , 'task'] = 1
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[2]-2501:OIR_concepts[2] ]], ignore_index= True)

IRO.loc[IRO_concepts[5]-2501:IRO_concepts[5] , 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO.loc[IRO_concepts[5]-2501:IRO_concepts[5] ]], ignore_index= True)
OIR.loc[OIR_concepts[5]-2501:OIR_concepts[5] ,'task'] = 3
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[5]-2501:OIR_concepts[5] ]], ignore_index= True)
IRO.loc[IRO_concepts[7]-2501:IRO_concepts[7] , 'task'] = 2
new_dataset = pd.concat([new_dataset, IRO.loc[IRO_concepts[7]-2501:IRO_concepts[7] ]], ignore_index= True)
OIR.loc[OIR_concepts[7]-2501:OIR_concepts[7] , 'task'] = 3
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[7]-2501:OIR_concepts[7] ]], ignore_index= True)
OIR.loc[OIR_concepts[4]-2501:OIR_concepts[4] , 'task'] = 1
new_dataset = pd.concat([new_dataset, OIR.loc[OIR_concepts[4]-2501:OIR_concepts[4] ]], ignore_index= True)

new_dataset.to_csv('./local_datasets/looploop_dataset.csv', index=False)
"""