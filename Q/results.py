import matplotlib.pyplot as plt
from math import *
import numpy as np

eps_trained_on = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 800, 1000, 1200, 1400, 1600]

sarsa_ave_of_aves = [22.237500000000004, 21.890500000000007, 22.86350000000001, 24.012999999999998, 26.288999999999998, 28.015, 33.8095, 33.311, 41.42049999999999, 46.39550000000001, 44.098500000000016, 53.39050000000001, 79.98000000000003, 77.44349999999999, 107.94550000000001, 116.15150000000003, 105.2775, 135.90999999999997, 178.24250000000006, 192.03849999999994, 270.7675, 256.22149999999993, 236.66800000000003, 243.40849999999995, 203.58899999999997, 285.80049999999994, 180.7345, 210.09049999999996, 234.83799999999997, 188.78800000000004, 208.19649999999993, 262.2125, 259.35300000000007, 330.65399999999994, 243.99449999999987]
# sarsa_ave_of_max = [51.17, 50.65, 55.78, 58.88, 65.56, 70.35, 89.91, 88.95, 107.51, 115.63, 106.13, 132.15, 206.8, 188.33, 263.25, 275.54, 229.92, 306.12, 409.6, 411.49, 644.53, 625.99, 542.93, 584.86, 492.59, 728.81, 376.52, 516.05, 589.45, 457.43, 486.29, 620.61, 642.92, 764.99, 530.16]
# sarsa_max_of_max = [94, 111, 113, 127, 149, 159, 351, 322, 358, 533, 381, 1040, 1895, 1900, 2041, 2635, 1681, 3663, 3232, 3861, 4621, 8598, 5326, 5906, 5768, 6900, 3832, 5405, 4431, 3905, 5028, 5484, 4075, 4609, 3392]   

Q_ave_of_aves = [22.0255, 22.085499999999993, 22.789999999999996, 25.374499999999998, 28.28749999999999, 33.86100000000001, 36.15550000000001, 43.94250000000001, 53.3855, 59.34800000000005, 68.64850000000003, 84.79199999999999, 100.61500000000004, 132.51300000000003, 155.89299999999997, 281.3889999999999, 351.3214999999998, 522.3505, 630.9970000000002, 862.4290000000001, 752.5745, 735.3575, 624.0795000000005, 580.7204999999999, 587.5980000000001, 666.365, 614.4735000000001, 676.354, 594.5985, 682.9044999999999, 603.2215, 565.4030000000004, 702.5755, 668.0314999999998, 546.6375000000003]
# Q_ave_of_max = [52.61, 50.91, 55.32, 61.46, 72.61, 89.25, 91.35, 117.66, 136.39, 154.0, 176.24, 228.4, 249.56, 347.73, 379.51, 696.26, 885.13, 1413.77, 1699.69, 2273.99, 1932.26, 1846.2, 1594.86, 1403.85, 1440.01, 1767.66, 1613.76, 1691.84, 1539.39, 1675.32, 1646.19, 1424.89, 1808.35, 1668.64, 1312.45]
# Q_max_of_max = [104, 97, 121, 118, 180, 228, 210, 303, 468, 465, 632, 1604, 1498, 2603, 2513, 3391, 4980, 5456, 7317, 6441, 5766, 4817, 7789, 4970, 5998, 7160, 6072, 5610, 6976, 8012, 8785, 7126, 5504, 5123, 5597]   

# Plot the graphs
plt.scatter(eps_trained_on, sarsa_ave_of_aves)
plt.xlabel('Number of episodes trained on')
plt.ylabel('Average reward')
plt.title('Sarsa: Ave rewards for agents trained on variang number of episode')
plt.show()
# plt.scatter(eps_trained_on, sarsa_ave_of_max)
# plt.xlabel('Number of episodes trained on')
# plt.ylabel('Average of max rewards')
# plt.title('Sarsa: Ave of max reward gained by agents trained on variang number of episode')
# plt.show()
# plt.scatter(eps_trained_on, sarsa_max_of_max)
# plt.xlabel('Number of episodes trained on')
# plt.ylabel('Max of max rewards')
# plt.title('Sarsa: Max reward gained by agents trained on variang number of episode')
# plt.show()

plt.scatter(eps_trained_on, Q_ave_of_aves)
plt.xlabel('Number of episodes trained on')
plt.ylabel('Average reward')
plt.title('Q-learing: Ave rewards for agents trained on variang number of episode')
plt.show()
# plt.scatter(eps_trained_on, Q_ave_of_max)
# plt.xlabel('Number of episodes trained on')
# plt.ylabel('Average of max rewards')
# plt.title('Q-learing: Ave of max reward gained by agents trained on variang number of episode')
# plt.show()
# plt.scatter(eps_trained_on, Q_max_of_max)
# plt.xlabel('Number of episodes trained on')
# plt.ylabel('Max of max rewards')
# plt.title('Q-learing: Max reward gained by agents trained on variang number of episode')
# plt.show()


# decays = [2, 5, 10, 20, 30, 50, 75, 100, 150]

# sarsa_d_ave_of_aves = [83.1755, 62.583500000000015, 109.14300000000003, 212.01750000000004, 309.472, 318.62149999999997, 159.09750000000008, 116.00199999999997, 54.308499999999995]
# sarsa_d_ave_of_max = [188.46, 133.12, 242.08, 472.7, 753.87, 797.45, 349.77, 260.1, 121.12]
# sarsa_d_max_of_max = [4167, 1236, 3221, 3816, 4487, 5718, 3809, 4631, 2077]

# Qd_ave_of_aves = [228.933, 466.973, 613.5550000000002, 614.2479999999999, 822.6549999999999, 662.4810000000001, 657.62, 489.0394999999998, 281.0465]
# Qd_ave_of_max = [631.76, 1184.49, 1543.26, 1595.95, 2114.81, 1619.32, 1690.52, 1204.94, 681.25]
# Qd_max_of_max = [6688, 5653, 6639, 5012, 6471, 6127, 5786, 8478, 3010]

# # Plot the graphs
# plt.scatter(decays, sarsa_d_ave_of_aves)
# plt.xlabel('Decay Rate')
# plt.ylabel('Average reward')
# plt.title('Sarsa: Ave rewards for agents trained on variang decay rates')
# plt.show()
# plt.scatter(decays, sarsa_d_ave_of_max)
# plt.xlabel('Decay Rate')
# plt.ylabel('Average of max rewards')
# plt.title('Sarsa: Ave of max reward gained by agents trained on variang decay rates')
# plt.show()
# plt.scatter(decays, sarsa_d_max_of_max)
# plt.xlabel('Decay Rate')
# plt.ylabel('Max of max rewards')
# plt.title('Sarsa: Max reward gained by agents trained on variang decay rates')
# plt.show()

# # Plot the graphs
# plt.scatter(decays, Qd_ave_of_aves)
# plt.xlabel('Decay Rate')
# plt.ylabel('Average reward')
# plt.title('Q-learing: Ave rewards for agents trained on variang decay rate')
# plt.show()
# plt.scatter(decays, Qd_ave_of_max)
# plt.xlabel('Decay Rate')
# plt.ylabel('Average of max rewards')
# plt.title('Q-learing: Ave of max reward gained by agents trained on variang decay rate')
# plt.show()
# plt.scatter(decays, Qd_max_of_max)
# plt.xlabel('Decay Rate')
# plt.ylabel('Max of max rewards')
# plt.title('Q-learing: Max reward gained by agents trained on variang decay rate')
# plt.show()