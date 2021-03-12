# Data visualization for research data
# DAA Research
# A 43 Ayush Kedia
# A 53 Mihir Goyenka
# Plotting data in different forms for visual perception for KNN search only

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

ax = plt.axes(projection='3d')

X2 = [5, 10, 15, 20, 25, 35, 50, 75, 100, 150]
X3 = [10, 25, 40, 55, 70, 85, 100]  # X3 -- X7 up to 100
# X4 = [10, 25, 40, 55, 70, 85, 100]
# X5 = [10, 25, 40, 55, 70, 85, 100]
# X6 = [10, 25, 40, 55, 70, 85, 100]
# X7 = [10, 25, 40, 55, 70, 85, 100]
X8 = [10, 25, 40, 55, 70, 85]
X9 = [10, 25, 40, 55, 70]  # X9 -- X13 up to 70
# X10 = [10, 25, 40, 55, 70]
# X11 = [10, 25, 40, 55, 70]
# X12 = [10, 25, 40, 55, 70]
# X13 = [10, 25, 40, 55, 70]
X14 = [10, 25, 40, 55]  # X14 -- X20 up to 55
# X15 = [10, 25, 40, 55]
# X16 = [10, 25, 40, 55]
# X17 = [10, 25, 40, 55]
# X18 = [10, 25, 40, 55]
# X19 = [10, 25, 40, 55]
# X20 = [10, 25, 40, 55]

# repeated Y required in wireframe plot
Y2 = [2] * 10
Y3 = [3] * 7
Y4 = [4] * 7
Y5 = [5] * 7
Y6 = [6] * 7
Y7 = [7] * 7
Y8 = [8] * 6
Y9 = [9] * 5
Y10 = [10] * 5
Y11 = [11] * 5
Y12 = [12] * 5
Y13 = [13] * 5
Y14 = [14] * 4
Y15 = [15] * 4
Y16 = [16] * 4
Y17 = [17] * 4
Y18 = [18] * 4
Y19 = [19] * 4
Y20 = [20] * 4

# E+10 is unit for each point
Z2 = [0.26449639000000000, 0.77530808000000000, 2.5879862700000000, 2.8112809100000000, 4.0315292400000000,
      6.8794890900000000, 12.053953980000000, 23.210090690000000, 36.325339050000000, 74.655691800000000]
Z3 = [1.9307090400000000, 8.8061613600000000, 17.969014150000000, 34.290850470000000,
      40.066042890000000, 44.445486880000000, 69.533130570000000]
Z4 = [1.9056698100000000, 8.9601958700000000, 21.842974980000000, 36.785940190000000,
      45.762190120000000, 58.033865670000000, 65.928921960000000]
Z5 = [2.0842700700000000, 8.7614840100000000, 21.714204620000000, 29.924213340000000,
      40.422580740000000, 57.692210150000000, 76.775731190000000]
Z6 = [2.0164624400000000, 6.3651478100000000, 17.602437080000000, 27.762563310000000,
      45.802987000000000, 63.334948350000000, 109.98702458000000]
Z7 = [1.2860187000000000, 6.8945119700000000, 14.982487560000000, 27.501525240000000, 36.217763490000000,
      48.425046660000000, 64.707589120000000]
Z8 = [1.9466994400, 8.2402750200, 18.5410387500, 39.4624817100, 47.8043069700, 59.6424911200]
Z9 = [2.08352812000, 9.43437011000, 21.7375868000, 39.8595042600, 55.2550058300]
Z10 = [1.94480242000, 11.3097048800, 25.0635900400, 48.6134208100, 67.5784028400]
Z11 = [2.01754232000, 11.2583594500, 27.7880860000, 71.6700884700, 76.1988252200]
Z12 = [3.08565563000, 14.6835042800, 30.7576537100, 53.6608687300, 87.1397389400]
Z13 = [2.30932720000, 12.4209596500, 45.8700668500, 79.0384821100, 134.602588500]
Z14 = [2.31640858000, 12.7391884500, 33.3065692500, 67.8884679300]
Z15 = [2.444868420000, 12.71764407000, 39.40109276000, 75.90308313000]
Z16 = [2.872620550000, 16.74325339000, 43.10131024000, 76.05085298000]
Z17 = [2.471265460000, 13.93660782000, 44.24587733000, 62.87427025000]
Z18 = [3.030601130000, 15.80153573000, 36.74592999000, 64.81570668000]
Z19 = [2.547995650000, 13.97969123000, 38.29544119000, 65.17860841000]
Z20 = [2.586963710000, 14.84726703000, 38.74705025000, 67.04268474000]

plt.title("KNN SEARCH TIME for Different Dimensions and Data Points")
ax.scatter(X2, 2, Z2, c='blue', marker='2')
ax.scatter(X3, 3, Z3, c='#ffff00', marker='2')
ax.scatter(X3, 4, Z4, c='#ff0066', marker='2')
ax.scatter(X3, 5, Z5, c='#1a8cff', marker='2')
ax.scatter(X3, 6, Z6, c='#000000', marker='2')
ax.scatter(X3, 7, Z7, c='red', marker='2')
ax.scatter(X8, 8, Z8, c='green', marker='2')
ax.scatter(X9, 9, Z9, c='yellow', marker='2')
ax.scatter(X9, 10, Z10, c='black', marker='2')
ax.scatter(X9, 11, Z11, c='cyan', marker='2')
ax.scatter(X9, 12, Z12, c='#ff33ff', marker='2')
ax.scatter(X9, 13, Z13, c='#1a6600', marker='2')
ax.scatter(X14, 14, Z14, c='#b30047', marker='2')
ax.scatter(X14, 15, Z15, c='#997300', marker='2')
ax.scatter(X14, 16, Z16, c='#000080', marker='2')
ax.scatter(X14, 17, Z17, c='#000000', marker='2')
ax.scatter(X14, 18, Z18, c='#e600e6', marker='2')
ax.scatter(X14, 19, Z19, c='#993333', marker='2')
ax.scatter(X14, 20, Z20, c='#33cc33', marker='2')

Z2 = np.array([Z2, Z2])
Z3 = np.array([Z3, Z3])
Z4 = np.array([Z4, Z4])
Z5 = np.array([Z5, Z5])
Z6 = np.array([Z6, Z6])
Z7 = np.array([Z7, Z7])
Z8 = np.array([Z8, Z8])
Z9 = np.array([Z9, Z9])
Z10 = np.array([Z10, Z10])
Z11 = np.array([Z11, Z11])
Z12 = np.array([Z12, Z12])
Z13 = np.array([Z13, Z13])
Z14 = np.array([Z14, Z14])
Z15 = np.array([Z15, Z15])
Z16 = np.array([Z16, Z16])
Z17 = np.array([Z17, Z17])
Z18 = np.array([Z18, Z18])
Z19 = np.array([Z19, Z19])
Z20 = np.array([Z20, Z20])

ax.plot_wireframe(X2, Y2, Z2, color='blue', linewidth=0.3)
ax.plot_wireframe(X3, Y3, Z3, color='#ffff00', linewidth=0.3)
ax.plot_wireframe(X3, Y4, Z4, color='#ff0066', linewidth=0.3)
ax.plot_wireframe(X3, Y5, Z5, color='#1a8cff', linewidth=0.3)
ax.plot_wireframe(X3, Y6, Z6, color='#000000', linewidth=0.3)
ax.plot_wireframe(X3, Y7, Z7, color='red', linewidth=0.3)
ax.plot_wireframe(X8, Y8, Z8, color='green', linewidth=0.3)
ax.plot_wireframe(X9, Y9, Z9, color='yellow', linewidth=0.3)
ax.plot_wireframe(X9, Y10, Z10, color='black', linewidth=0.3)
ax.plot_wireframe(X9, Y11, Z11, color='cyan', linewidth=0.3)
ax.plot_wireframe(X9, Y12, Z12, color='#ff33ff', linewidth=0.3)
ax.plot_wireframe(X9, Y13, Z13, color='#1a6600', linewidth=0.3)
ax.plot_wireframe(X14, Y14, Z14, color='#b30047', linewidth=0.3)
ax.plot_wireframe(X14, Y15, Z15, color='#997300', linewidth=0.3)
ax.plot_wireframe(X14, Y16, Z16, color='#000080', linewidth=0.3)
ax.plot_wireframe(X14, Y17, Z17, color='#000000', linewidth=0.3)
ax.plot_wireframe(X14, Y18, Z18, color='#e600e6', linewidth=0.3)
ax.plot_wireframe(X14, Y19, Z19, color='#993333', linewidth=0.3)
ax.plot_wireframe(X14, Y20, Z20, color='#33cc33', linewidth=0.3)

ax.set_xlabel('No. of Data points(in 1000s)')
ax.set_ylabel('Dimensions')
ax.set_zlabel('Time in E+10 ns')

plt.show()
