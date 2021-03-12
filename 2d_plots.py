# 2d plot comparisons for individual dimensions of knn search and parallel search

import matplotlib.pyplot as plt

fig = plt.figure()
fig.patch.set_facecolor('#ccffff')
fig.patch.set_alpha(0.025)

ax1 = plt.subplot2grid((5, 4), (0, 0))  # for dim = 2
ax2 = plt.subplot2grid((5, 4), (0, 1))
ax3 = plt.subplot2grid((5, 4), (0, 2))
ax4 = plt.subplot2grid((5, 4), (0, 3))
ax5 = plt.subplot2grid((5, 4), (1, 0))
ax6 = plt.subplot2grid((5, 4), (1, 1))
ax7 = plt.subplot2grid((5, 4), (1, 2))
ax8 = plt.subplot2grid((5, 4), (1, 3))
ax9 = plt.subplot2grid((5, 4), (2, 0))
ax10 = plt.subplot2grid((5, 4), (2, 1))
ax11 = plt.subplot2grid((5, 4), (2, 2))
ax12 = plt.subplot2grid((5, 4), (2, 3))
ax13 = plt.subplot2grid((5, 4), (3, 0))
ax14 = plt.subplot2grid((5, 4), (3, 1))
ax15 = plt.subplot2grid((5, 4), (3, 2))
ax16 = plt.subplot2grid((5, 4), (3, 3))
ax17 = plt.subplot2grid((5, 4), (4, 0))
ax18 = plt.subplot2grid((5, 4), (4, 1))
ax19 = plt.subplot2grid((5, 4), (4, 2))  # for dim = 20

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
Zp10 = [0.62005650000, 5.5516725700, 13.641144380, 21.971811820, 34.633833290]
Zp11 = [0.63079490000, 6.0406935900, 14.595701310, 44.409485330, 42.051630880]
Zp12 = [1.0325194500, 6.7567322900, 16.349422080, 28.624714720, 46.139401680]
Zp13 = [1.2343049400, 6.4844150300, 24.147242250, 44.498562470, 73.216991100]
Zp14 = [0.80219469000, 7.2069365800, 18.136142710, 32.297809110]
Zp15 = [1.1068116500, 6.8335651500, 18.426735540, 32.965958950]
Zp16 = [2.0372600700, 8.5461623100, 18.670175220, 34.862428720]
Zp17 = [0.91409304000, 7.1964435400, 18.654472160, 36.329342740]
Zp18 = [1.7125994900, 10.116063860, 21.203008550, 39.142287060]
Zp19 = [0.88573471000, 7.3999818900, 19.842115340, 35.885930790]
Zp20 = [0.85153915000, 7.5576028400, 24.524720380, 36.738903190]

l1 = ax1.plot(X2, Z2, color='r')[0]
l2 = ax1.plot(X2, Zp2, color='b')[0]
ax2.plot(X3, Z3, color='r')
ax2.plot(X3, Zp3, color='b')
ax3.plot(X3, Z4, color='r')
ax3.plot(X3, Zp4, color='b')
ax4.plot(X3, Z5, color='r')
ax4.plot(X3, Zp5, color='b')
ax5.plot(X3, Z6, color='r')
ax5.plot(X3, Zp6, color='b')
ax6.plot(X3, Z7, color='r')
ax6.plot(X3, Zp7, color='b')
ax7.plot(X8, Z8, color='r')
ax7.plot(X8, Zp8, color='b')
ax8.plot(X9, Z9, color='r')
ax8.plot(X9, Zp9, color='b')
ax9.plot(X9, Z10, color='r')
ax9.plot(X9, Zp10, color='b')
ax10.plot(X9, Z11, color='r')
ax10.plot(X9, Zp11, color='b')
ax11.plot(X9, Z12, color='r')
ax11.plot(X9, Zp12, color='b')
ax12.plot(X9, Z13, color='r')
ax12.plot(X9, Zp13, color='b')
ax13.plot(X14, Z14, color='r')
ax13.plot(X14, Zp14, color='b')
ax14.plot(X14, Z15, color='r')
ax14.plot(X14, Zp15, color='b')
ax15.plot(X14, Z16, color='r')
ax15.plot(X14, Zp16, color='b')
ax16.plot(X14, Z17, color='r')
ax16.plot(X14, Zp17, color='b')
ax17.plot(X14, Z18, color='r')
ax17.plot(X14, Zp18, color='b')
ax18.plot(X14, Z19, color='r')
ax18.plot(X14, Zp19, color='b')
ax19.plot(X14, Z20, color='r')
ax19.plot(X14, Zp20, color='b')

ax1.set_title("Dimension = 2"),  # ax1.set_facecolor('#ccffff')
ax2.set_title("Dimension = 3"),  # ax2.set_facecolor('#ccffff')
ax3.set_title("Dimension = 4"),  # ax3.set_facecolor('#ccffff')
ax4.set_title("Dimension = 5"),  # ax4.set_facecolor('#ccffff')
ax5.set_title("Dimension = 6"),  # ax5.set_facecolor('#ccffff')
ax6.set_title("Dimension = 7"),  # ax6.set_facecolor('#ccffff')
ax7.set_title("Dimension = 8"),  # ax7.set_facecolor('#ccffff')
ax8.set_title("Dimension = 9"),  # ax8.set_facecolor('#ccffff')
ax9.set_title("Dimension = 10"),  # ax9.set_facecolor('#ccffff')
ax10.set_title("Dimension = 11"),  # ax10.set_facecolor('#ccffff')
ax11.set_title("Dimension = 12"),  # ax11.set_facecolor('#ccffff')
ax12.set_title("Dimension = 13"),  # ax12.set_facecolor('#ccffff')
ax13.set_title("Dimension = 14"),  # ax13.set_facecolor('#ccffff')
ax14.set_title("Dimension = 15"),  # ax14.set_facecolor('#ccffff')
ax15.set_title("Dimension = 16"),  # ax15.set_facecolor('#ccffff')
ax16.set_title("Dimension = 17"),  # ax16.set_facecolor('#ccffff')
ax17.set_title("Dimension = 18"),  # ax17.set_facecolor('#ccffff')
ax18.set_title("Dimension = 19"),  # ax18.set_facecolor('#ccffff')
ax19.set_title("Dimension = 20"),  # ax19.set_facecolor('#ccffff')

# Set common labels for axes for all sub plots
fig.text(0.5, 0.04, '<------No. of data points processed------>', ha='center', va='center', fontsize=15)
fig.text(0.06, 0.5, '<------Time in E+10 ns------>', ha='center', va='center', rotation='vertical', fontsize=15)
fig.suptitle("2D Plot for KNN Search & KNN Parallel Search Time Comparison", fontsize=20)
line_labels = ["knn", "parallel_knn"]
fig.legend([l1, l2], line_labels, loc="lower right", borderaxespad=0.1, title='Legend', fontsize=20)
plt.subplots_adjust(hspace=0.75)

plt.show()
