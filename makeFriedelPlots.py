import numpy as np

from matplotlib import pyplot as plt

from hexrd.xrd import rotations as rot
from hexrd.xrd import crystallography as xtl

r2d = 180. / np.pi
d2r = np.pi / 180.

eta0 = np.arange(-180, 180.05, 0.05)
tth0 = 10*np.ones_like(eta0)

chi  = [0, 1, 5, 10, 20, 30]

lc = ['k', 'r', 'g', 'b', 'c', 'm', 'y']
lt = [lc[i] + '-' for i in range(len(lc))]

set11 = eta0 <= -90
set12 = np.logical_and(eta0 > -90, eta0 <= 90)
set13 = eta0 > 90

lw = 2
fig1 = plt.figure(1); axes1 = fig1.gca()
fig2 = plt.figure(2); axes2 = fig2.gca()
for i in range(len(chi)):
    ome1, eta1 = xtl.getFriedelPair(tth0, eta0,
                                    chiTilt=chi[i],
                                    units='degrees',
                                    convention='hexrd')
    axes1.plot(d2r*eta0[set11], ome1[set11], lt[i],
             label='$\chi$=%d$^\circ$' %chi[i], linewidth=lw)
    axes1.plot(d2r*eta0[set12], ome1[set12], lt[i],
             label='$\chi$=%d$^\circ$' %chi[i], linewidth=lw)
    axes1.plot(d2r*eta0[set13], ome1[set13], lt[i],
             label='$\chi$=%d$^\circ$' %chi[i], linewidth=lw)
    eta = -rot.angularDifference(d2r*eta0, eta1)*np.sign(np.sin(eta1))
    set1 = np.logical_and(eta0 < -90., eta <= 0.)
    set2 = np.logical_and(eta0 <  90., eta >= 0.)
    set3 = np.logical_and(np.logical_and(eta0 > -90., eta0 < 90.), eta <= 0.)
    set4 = eta0 > 90.
    axes2.plot(d2r*eta0[set1], eta[set1], lt[i], linewidth=lw)
    axes2.plot(d2r*eta0[set2], eta[set2], lt[i], linewidth=lw)
    axes2.plot(d2r*eta0[set3], eta[set3], lt[i], linewidth=lw)
    axes2.plot(d2r*eta0[set4], eta[set4], lt[i], linewidth=lw)

axes1.grid(True)
axes1.axis('tight')

axes1.set_xlabel('starting azimuth, $\eta_0$')
axes1.set_ylabel('minimum $\Delta\omega_{FP}$')

axes1.set_xticks(np.arange(-180, 225, 45)*d2r)
axes1.set_xticklabels(['$-\pi$', '$-3\pi/4$', '$-\pi/2$',
                      '$-\pi/4$', '$0$', '$\pi/4$',
                      '$\pi/2$', '$3\pi/4$', '$\pi$'])
axes1.set_yticks(np.r_[-180, -120, -60, -tth0[0], 0, tth0[0], 60, 120, 180]*d2r)
axes1.set_yticklabels(['$-\pi$', '$-2\pi/3$', '$-\pi/3$',
                      '$-2\\theta$', '$0$', '$2\\theta$',
                      '$\pi/3$', '$2\pi/3$', '$\pi$'])
fig1.show()

axes2.set_xlabel('starting azimuth, $\eta_0$')
axes2.set_ylabel('azimuthal difference, $\eta_{FP} - \pi$')

axes2.set_xlim(-np.pi, np.pi)
axes2.set_ylim(-np.pi/6., np.pi/6.)

axes2.set_xticks(np.arange(-180, 225, 45)*d2r)
axes2.set_xticklabels(['$-\pi$', '$-3\pi/4$', '$-\pi/2$',
                      '$-\pi/4$', '$0$', '$\pi/4$',
                      '$\pi/2$', '$3\pi/4$', '$\pi$'])
axes2.set_yticks(np.arange(-3, 4)*tth0[0]*d2r)
axes2.set_yticklabels(['-6\\theta$', '-4\\theta', '-2\\theta', '$0$', '2\\theta', '4\\theta', '6\\theta'])
axes2.grid(True)

fig2.show()
