import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.ticker as mtick
from matplotlib import cm  # Colormaps
import matplotlib.gridspec as gridspec
import seaborn as sns

from tutorial_utils.physical_methods import laminar_flow_wall_coordinates

plt.rc('mathtext', fontset="cm")

colors = cm.tab10(np.linspace(0, 1, 10))
markers = ['o', '^', 's', 'd', 'P', '*', 'v', 'D']
line_width = 1


def get_laminar_data_for_plotting():
    possible_re_tau = [20, 40, 60, 80, 100]
    Y_plus_list = []
    U_plus_list = []

    for re_tau in possible_re_tau:
        Y_plus = []
        U_plus = []

        for y_plus in np.linspace(start=0.01, stop=re_tau, num=100):
            u_plus = laminar_flow_wall_coordinates(y_plus, re_tau)
            Y_plus.append(y_plus)
            U_plus.append(u_plus)

        Y_plus_list.append(Y_plus)
        U_plus_list.append(U_plus)

    return U_plus_list, Y_plus_list, possible_re_tau


def plot_laminar_flow(U_plus, Y_plus, Re_tau):
    fig, ax1 = plt.subplots(figsize=(16, 10))

    left, bottom, width, height = [0.57, 0.17, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])

    for i in range(len(Re_tau)):
        ax1.scatter(Y_plus[i], U_plus[i], alpha=1, s=40, linewidth=line_width,
                    facecolors='none', color=colors[i], marker=markers[i],
                    label=r'$Re_\tau= $'+r'${}$'.format(Re_tau[i]))

        ax2.scatter(Y_plus[i], U_plus[i], alpha=1, s=40, linewidth=line_width,
                    facecolors='none', color=colors[i], marker=markers[i])

    font = font_manager.FontProperties(family='DejaVu Sans', weight='roman',
                                       style='normal', size='large',
                                       stretch='ultra-condensed')

    ax1.legend(bbox_to_anchor=(0., 0.99), loc='upper left', edgecolor='white',
               framealpha=0, prop=font, borderaxespad=0.)
    ax1.set_xlabel(r"$y^+$", size='xx-large', fontweight='black')
    ax1.set_ylabel(r'$u^+$', size='xx-large')
    # ax1.set_ylim([5,14])
    ax1.grid(False)
    ax1.tick_params(axis='x', labelsize='large')
    ax1.tick_params(axis='y', labelsize='large')
    ax1.spines['bottom'].set_color('black')
    ax1.spines['top'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax2.spines['bottom'].set_color('black')
    ax2.spines['top'].set_color('black')
    ax2.spines['left'].set_color('black')
    ax2.spines['right'].set_color('black')
    ax2.set_xlabel(r"$\log \ y^+ $", fontsize='large')
    ax2.set_ylabel(r'$u^+$', fontsize='large')
    ax2.set_xscale('log')
    ax2.tick_params(axis='x', labelsize='medium')
    ax2.tick_params(axis='y', labelsize='medium')
    # ax2.set_xlim([0.01,1])
    # ax2.set_ylim([5,14])
    ax2.grid(False)

    plt.savefig('tutorial_plots/laminar_flow_wall_coordinates.pdf',
                format='pdf', dpi=1200)
    plt.show()
