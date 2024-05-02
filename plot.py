# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 19:27:52 2023

@author: tamar
"""

import matplotlib.pyplot as plt

N = 11
probabilities = [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
probabilities2 = [1,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
plt.figure(figsize=(5,5))
# Plotting the vertical line
plt.plot(range(N), probabilities, linestyle='-', label='Concept 1', color='green')
plt.plot(range(N), probabilities2, linestyle='-', label='Concept 2')
# Set labels and title
plt.xticks(range(0,11))
plt.xlabel('Event')
plt.ylabel('Probabilities (0 to 1)')
plt.legend()  # Show legend

# Show the plot
plt.show()