# comparative plot of simple knn and parallel knn search time
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

# E+10 is unit for each point in time axis

# simple knn search data
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

# parallel search data
Zp2 = [0.30778027000000000, 0.43506056000000000, 1.6340325300000000, 1.1337077900000000, 1.6477385700000000,
       3.3005987600000000, 6.4773454500000000, 12.582768060000000, 19.331853760000000, 37.845075990000000]
Zp3 = [1.4915146700000000, 1.8202718900000000, 5.1245091800000000, 7.9723129300000000,
       11.369732350000000, 14.937499370000000, 19.273802530000000]
Zp4 = [0.92177691000000000, 3.0640954800000000, 5.2199420900000000, 8.1805005300000000,
       11.730190180000000, 15.841541340000000, 20.640580010000000]
Zp5 = [0.56217170000000000, 3.3326192500000000, 5.8607671800000000,
       9.6498660500000000, 13.553231500000000, 17.719741480000000, 23.297821160000000]
Zp6 = [0.56434767000000000, 3.5686048000000000, 7.1031185000000000, 11.896982450000000,
       16.221114490000000, 21.833924750000000, 28.502212980000000]
Zp7 = [0.76117856000000000, 4.0409917800000000, 8.6726112200000000,
       14.066333220000000, 19.818378250000000, 27.095370880000000, 34.786229260000000]
Zp8 = [0.68764739000, 3.8132355600, 9.0401124000, 16.048932890, 24.444279540, 30.864233370]
Zp9 = [0.64111481000, 4.7062703200, 11.007726890, 19.198834710, 28.914611040]
Zp10 = [0.62005650000,  5.5516725700, 13.641144380, 21.971811820, 34.633833290]
Zp11 = [0.63079490000, 6.0406935900, 14.595701310, 44.409485330, 42.051630880]
Zp12 = [1.0325194500, 6.7567322900, 16.349422080, 28.624714720, 46.139401680]
Zp13 = [1.2343049400, 6.4844150300, 24.147242250, 44.498562470, 73.216991100]
Zp14 = [0.80219469000, 7.2069365800, 18.136142710, 32.297809110]
Zp15 = [1.1068116500, 6.8335651500, 18.426735540, 32.965958950]
Zp16 = [2.0372600700,  8.5461623100, 18.670175220, 34.862428720]
Zp17 = [0.91409304000, 7.1964435400, 18.654472160, 36.329342740]
Zp18 = [1.7125994900, 10.116063860, 21.203008550, 39.142287060]
Zp19 = [0.88573471000, 7.3999818900, 19.842115340, 35.885930790]
Zp20 = [0.85153915000, 7.5576028400, 24.524720380, 36.738903190]

plt.title("PARALLEL KNN SEARCH and KNN SEARCH TIME comparison for Different Dimensions and Data Points")

# . for parallel search time
ax.scatter(X2, 2, Zp2, c='blue', marker='d', label='Parallel')
ax.scatter(X3, 3, Zp3, c='#ffff00', marker='d')
ax.scatter(X3, 4, Zp4, c='#ff0066', marker='d')
ax.scatter(X3, 5, Zp5, c='#1a8cff', marker='d')
ax.scatter(X3, 6, Zp6, c='#000000', marker='d')
ax.scatter(X3, 7, Zp7, c='red', marker='d')
ax.scatter(X8, 8, Zp8, c='green', marker='d')
ax.scatter(X9, 9, Zp9, c='yellow', marker='d')
ax.scatter(X9, 10, Zp10, c='black', marker='d')
ax.scatter(X9, 11, Zp11, c='cyan', marker='d')
ax.scatter(X9, 12, Zp12, c='#ff33ff', marker='d')
ax.scatter(X9, 13, Zp13, c='#1a6600', marker='d')
ax.scatter(X14, 14, Zp14, c='#b30047', marker='d')
ax.scatter(X14, 15, Zp15, c='#997300', marker='d')
ax.scatter(X14, 16, Zp16, c='#000080', marker='d')
ax.scatter(X14, 17, Zp17, c='#000000', marker='d')
ax.scatter(X14, 18, Zp18, c='#e600e6', marker='d')
ax.scatter(X14, 19, Zp19, c='#993333', marker='d')
ax.scatter(X14, 20, Zp20, c='#33cc33', marker='d')

# simple knn search scatter plot
ax.scatter(X2, 2, Z2, c='b', marker='2', label='simple knn')
ax.scatter(X3, 3, Z3, c='#ffff00', marker='2')
ax.scatter(X3, 4, Z4, c='#ff0066', marker='2')
ax.scatter(X3, 5, Z5, c='#1a8cff', marker='2')
ax.scatter(X3, 6, Z6, c='#000000', marker='2')
ax.scatter(X3, 7, Z7, c='r', marker='2')
ax.scatter(X8, 8, Z8, c='g', marker='2')
ax.scatter(X9, 9, Z9, c='y', marker='2')
ax.scatter(X9, 10, Z10, c='k', marker='2')
ax.scatter(X9, 11, Z11, c='c', marker='2')
ax.scatter(X9, 12, Z12, c='#ff33ff', marker='2')
ax.scatter(X9, 13, Z13, c='#1a6600', marker='2')
ax.scatter(X14, 14, Z14, c='#b30047', marker='2')
ax.scatter(X14, 15, Z15, c='#997300', marker='2')
ax.scatter(X14, 16, Z16, c='#000080', marker='2')
ax.scatter(X14, 17, Z17, c='#000000', marker='2')
ax.scatter(X14, 18, Z18, c='#e600e6', marker='2')
ax.scatter(X14, 19, Z19, c='#993333', marker='2')
ax.scatter(X14, 20, Z20, c='#33cc33', marker='2')

# parallel data
Zp2 = np.array([Zp2, Zp2])
Zp3 = np.array([Zp3, Zp3])
Zp4 = np.array([Zp4, Zp4])
Zp5 = np.array([Zp5, Zp5])
Zp6 = np.array([Zp6, Zp6])
Zp7 = np.array([Zp7, Zp7])
Zp8 = np.array([Zp8, Zp8])
Zp9 = np.array([Zp9, Zp9])
Zp10 = np.array([Zp10, Zp10])
Zp11 = np.array([Zp11, Zp11])
Zp12 = np.array([Zp12, Zp12])
Zp13 = np.array([Zp13, Zp13])
Zp14 = np.array([Zp14, Zp14])
Zp15 = np.array([Zp15, Zp15])
Zp16 = np.array([Zp16, Zp16])
Zp17 = np.array([Zp17, Zp17])
Zp18 = np.array([Zp18, Zp18])
Zp19 = np.array([Zp19, Zp19])
Zp20 = np.array([Zp20, Zp20])

# simple knn data
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

# parallel line plot
ax.plot_wireframe(X2, Y2, Zp2, color='blue', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X3, Y3, Zp3, color='#ffff00', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X3, Y4, Zp4, color='#ff0066', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X3, Y5, Zp5, color='#1a8cff', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X3, Y6, Zp6, color='#000000', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X3, Y7, Zp7, color='red', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X8, Y8, Zp8, color='green', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X9, Y9, Zp9, color='yellow', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X9, Y10, Zp10, color='black', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X9, Y11, Zp11, color='cyan', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X9, Y12, Zp12, color='#ff33ff', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X9, Y13, Zp13, color='#1a6600', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X14, Y14, Zp14, color='#b30047', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X14, Y15, Zp15, color='#997300', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X14, Y16, Zp16, color='#000080', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X14, Y17, Zp17, color='#000000', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X14, Y18, Zp18, color='#e600e6', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X14, Y19, Zp19, color='#993333', linewidth=0.5, linestyle='dotted')
ax.plot_wireframe(X14, Y20, Zp20, color='#33cc33', linewidth=0.5, linestyle='dotted')

# simple knn line plot
ax.plot_wireframe(X2, Y2, Z2, color='blue', linewidth=0.2)
ax.plot_wireframe(X3, Y3, Z3, color='#ffff00', linewidth=0.2)
ax.plot_wireframe(X3, Y4, Z4, color='#ff0066', linewidth=0.2)
ax.plot_wireframe(X3, Y5, Z5, color='#1a8cff', linewidth=0.2)
ax.plot_wireframe(X3, Y6, Z6, color='#000000', linewidth=0.2)
ax.plot_wireframe(X3, Y7, Z7, color='red', linewidth=0.2)
ax.plot_wireframe(X8, Y8, Z8, color='green', linewidth=0.2)
ax.plot_wireframe(X9, Y9, Z9, color='yellow', linewidth=0.2)
ax.plot_wireframe(X9, Y10, Z10, color='black', linewidth=0.2)
ax.plot_wireframe(X9, Y11, Z11, color='cyan', linewidth=0.2)
ax.plot_wireframe(X9, Y12, Z12, color='#ff33ff', linewidth=0.2)
ax.plot_wireframe(X9, Y13, Z13, color='#1a6600', linewidth=0.2)
ax.plot_wireframe(X14, Y14, Z14, color='#b30047', linewidth=0.2)
ax.plot_wireframe(X14, Y15, Z15, color='#997300', linewidth=0.2)
ax.plot_wireframe(X14, Y16, Z16, color='#000080', linewidth=0.2)
ax.plot_wireframe(X14, Y17, Z17, color='#000000', linewidth=0.2)
ax.plot_wireframe(X14, Y18, Z18, color='#e600e6', linewidth=0.2)
ax.plot_wireframe(X14, Y19, Z19, color='#993333', linewidth=0.2)
ax.plot_wireframe(X14, Y20, Z20, color='#33cc33', linewidth=0.2)

ax.set_xlabel('No. of Data points(in 1000s)')
ax.set_ylabel('Dimensions')
ax.set_zlabel('Time in E+10 ns')

plt.legend(loc='lower left', ncol=2, fontsize=10)
plt.show()
