import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import cm  # Colormaps
import seaborn as sns # noqa
from sympy import sqrt

from tutorial_utils.physical_methods import laminar_flow_wall_coordinates

plt.rc('mathtext', fontset="cm")

colors = cm.tab10(np.linspace(0, 1, 10))
markers = ['o', '^', 's', 'd', 'P', '*', 'v', 'D']
line_width = 1


def _get_laminar_data_for_plotting():
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


def plot_laminar_flow_renormalized(U_plus, Y_plus, Re_tau, exponents_dict):

    y_plus_exponent = exponents_dict["y+"]["Re_tau"]
    u_plus_exponent = exponents_dict["u+"]["Re_tau"]

    fig, ax1 = plt.subplots(figsize=(16, 10))

    left, bottom, width, height = [0.57, 0.17, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])

    for i in range(len(Re_tau)):
        ax1.scatter(np.array(Y_plus[i]) * (Re_tau[i] ** y_plus_exponent),
                    np.array(U_plus[i]) * (Re_tau[i] ** u_plus_exponent),
                    alpha=1, s=40, linewidth=line_width, facecolors='none',
                    color=colors[i], marker=markers[i],
                    label=r'$Re_\tau= $'+r'${}$'.format(Re_tau[i]))

        ax2.scatter(np.array(Y_plus[i]) * (Re_tau[i] ** y_plus_exponent),
                    np.array(U_plus[i]) * (Re_tau[i] ** u_plus_exponent),
                    alpha=1, s=40, linewidth=line_width, facecolors='none',
                    color=colors[i], marker=markers[i])

    font = font_manager.FontProperties(family='DejaVu Sans', weight='roman',
                                       style='normal', size='large',
                                       stretch='ultra-condensed')

    ax1.legend(bbox_to_anchor=(0., 0.99), loc='upper left', edgecolor='white',
               framealpha=0, prop=font, borderaxespad=0.)
    ax1.set_xlabel(r"$y^+ \times Re_\tau^{\xi_2^{(1)}}$", size='xx-large',
                   fontweight='black')
    ax1.set_ylabel(r'$u^+ \times Re_\tau^{\xi_1}$', size='xx-large')
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

    plt.savefig(
        'tutorial_plots/laminar_flow_wall_coordinates_renormalized.pdf',
        format='pdf', dpi=1200
    )
    plt.show()


def _get_extreme_re_data_for_plotting():

    dfSB2M300k_M = pd.read_csv(
        "../Data/McKeon_original_data/Re2300000.txt", sep="\t", header=None,
        names=["datapoint", "y/R", "y+", "U+", "(U+-Ucl+)"])
    dfSB3M_M = pd.read_csv(
        "../Data/McKeon_original_data/Re3000000.txt", sep="\t", header=None,
        names=["datapoint", "y/R", "y+", "U+", "(U+-Ucl+)"])
    dfSB4M_M = pd.read_csv(
        "../Data/McKeon_original_data/Re4000000.txt", sep="\t", header=None,
        names=["datapoint", "y/R", "y+", "U+", "(U+-Ucl+)"])
    dfSB6M_M = pd.read_csv(
        "../Data/McKeon_original_data/Re6000000.txt", sep="\t", header=None,
        names=["datapoint", "y/R", "y+", "U+", "(U+-Ucl+)"])
    dfSB7M_M = pd.read_csv(
        "../Data/McKeon_original_data/Re7000000.txt", sep="\t", header=None,
        names=["datapoint", "y/R", "y+", "U+", "(U+-Ucl+)"])
    dfSB10M_M = pd.read_csv(
        "../Data/McKeon_original_data/Re10000000.txt", sep="\t", header=None,
        names=["datapoint", "y/R", "y+", "U+", "(U+-Ucl+)"])
    dfSB13M_M = pd.read_csv(
        "../Data/McKeon_original_data/Re13000000.txt", sep="\t", header=None,
        names=["datapoint", "y/R", "y+", "U+", "(U+-Ucl+)"])

    dfs = [dfSB2M300k_M, dfSB3M_M, dfSB4M_M, dfSB6M_M,
           dfSB7M_M, dfSB10M_M, dfSB13M_M]
    Re_tau = []
    Y_plus = []
    U_plus = []
    possible_re_tau = [4.229500e+004, 5.453000e+004, 7.647800e+004, 1.022e+005,
                       1.279200e+005, 1.657e+005, 2.169800e+005]

    for i in range(len(possible_re_tau)):
        possible_y = dfs[i]["y+"].values
        possible_u = dfs[i]["U+"].values
        Y_plus.append(possible_y)
        U_plus.append(possible_u)
        Re_tau.append(possible_re_tau[i])

    return U_plus, Y_plus, Re_tau


def plot_extreme_re_flow():

    U_plus, Y_plus, Re_tau = _get_extreme_re_data_for_plotting()

    fig, ax1 = plt.subplots(figsize=(16, 10))

    left, bottom, width, height = [0.57, 0.17, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])

    for i in range(len(Re_tau)):
        ax1.scatter(Y_plus[i], U_plus[i], alpha=1, s=40, linewidth=line_width,
                    facecolors='none', color=colors[i], marker=markers[i],
                    label=r'$Re_\tau= $'+r'${:.2e}$'.format(Re_tau[i]))

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

    plt.savefig('tutorial_plots/extreme_re_flow_wall_coordinates.pdf',
                format='pdf', dpi=1200)
    plt.show()


def plot_extreme_re_flow_renormalized(exponents_dict):

    U_plus, Y_plus, Re_tau = _get_extreme_re_data_for_plotting()
    y_plus_exponent = exponents_dict["y+"]["Re_tau"]
    u_plus_exponent = exponents_dict["u+"]["Re_tau"]

    fig, ax1 = plt.subplots(figsize=(16, 10))

    left, bottom, width, height = [0.57, 0.17, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])

    colors = cm.tab10(np.linspace(0, 1, 10))
    markers = ['o', '^', 's', 'd', 'P', '*', 'v', 'D']
    line_width = 1

    for i in range(len(Re_tau)):
        ax1.scatter(np.array(Y_plus[i]) * (Re_tau[i] ** y_plus_exponent),
                    np.array(U_plus[i]) * (Re_tau[i] ** u_plus_exponent),
                    alpha=1, s=40, linewidth=line_width, facecolors='none',
                    color=colors[i], marker=markers[i],
                    label=r'$Re_\tau= $'+r'${:.2e}$'.format(Re_tau[i]))

        ax2.scatter(np.array(Y_plus[i]) * (Re_tau[i] ** y_plus_exponent),
                    np.array(U_plus[i]) * (Re_tau[i] ** u_plus_exponent),
                    alpha=1, s=40, linewidth=line_width, facecolors='none',
                    color=colors[i], marker=markers[i])

    font = font_manager.FontProperties(family='DejaVu Sans', weight='roman',
                                       style='normal', size='large',
                                       stretch='ultra-condensed')

    ax1.legend(bbox_to_anchor=(0., 0.99), loc='upper left', edgecolor='white',
               framealpha=0, prop=font, borderaxespad=0.)
    ax1.set_xlabel(r"$y^+ / Re_\tau^{1.16}$",
                   size='xx-large', fontweight='black')
    ax1.set_ylabel(r'$u^+ / Re_\tau^{0.09}$',
                   size='xx-large', fontweight='black')
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
    ax2.set_xlabel(r"$\log \ \left( y^+ / Re_\tau^{1.16} \right)$",
                   fontsize='large')
    ax2.set_ylabel(r'$u^+ / Re_\tau^{0.09}$',
                   fontsize='large')
    ax2.set_xscale('log')
    ax2.tick_params(axis='x', labelsize='medium')
    ax2.tick_params(axis='y', labelsize='medium')
    # ax2.set_xlim([0.01,1])
    # ax2.set_ylim([5,14])
    ax2.grid(False)

    plt.savefig(
        'tutorial_plots/extreme_re_flow_wall_coordinates_renormalized.pdf',
        format='pdf', dpi=1200)

    plt.show()


def _get_widom_scaling_data_for_plotting():

    # Amount of r points
    Nr = 30
    # Amount of j points
    Nj = 5
    # Minimum for r points
    mr = 0.01
    # Maximum for r points
    Mr = 20
    # Minimum for j points
    mj = 0.1
    # Maximum for j points
    Mj = 1

    u = 10

    def phi(r, j, u):
        return (r/(u*(-27*j/(2*u) +
                sqrt(729*j**2/u**2 +
                108*r**3/u**3)/2)**(1/3)) -
                (-27*j/(2*u) + sqrt(729*j**2/u**2 +
                 108*r**3/u**3)/2)**(1/3)/3)

    possible_r = np.linspace(mr, Mr, Nr)
    possible_j = np.linspace(mj, Mj, Nj)

    R_ = []
    J_ = []
    Phi_ = []

    for j in possible_j:
        R__ = []
        Phi__ = []
        for r in possible_r:
            R__.append(r)
            Phi__.append(phi(r, j, u))

        R_.append(R__)
        Phi_.append(Phi__)
        J_.append(j)

    return R_, J_, Phi_


def plot_widom_scaling():

    r, j, phi = _get_widom_scaling_data_for_plotting()

    fig, ax1 = plt.subplots(figsize=(16, 10))

    for i in range(len(j)):
        ax1.scatter(r[i], phi[i], alpha=1, s=40, linewidth=line_width,
                    facecolors='none', color=colors[i], marker=markers[i],
                    label=r'$j = $'+r'${:.2f}$'.format(j[i]))

    font = font_manager.FontProperties(family='DejaVu Sans', weight='roman',
                                       style='normal', size='large',
                                       stretch='ultra-condensed')

    ax1.legend(bbox_to_anchor=(0.90, 0.99), loc='upper left',
               edgecolor='white', framealpha=0, prop=font, borderaxespad=0.)
    ax1.set_xlabel(r"$r$", size='xx-large', fontweight='black')
    ax1.set_ylabel(r'$\phi$', size='xx-large')
    # ax1.set_ylim([5,14])
    ax1.grid(False)
    ax1.tick_params(axis='x', labelsize='large')
    ax1.tick_params(axis='y', labelsize='large')
    ax1.spines['bottom'].set_color('black')
    ax1.spines['top'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['right'].set_color('black')

    plt.savefig('tutorial_plots/widom_scaling.pdf',
                format='pdf', dpi=1200)
    plt.show()


def plot_widom_scaling_renormalized(exponents_dict):

    r, j, phi = _get_widom_scaling_data_for_plotting()

    fig, ax1 = plt.subplots(figsize=(16, 10))

    for i in range(len(j)):
        ax1.scatter(np.array(r[i]) * (j[i] ** exponents_dict["r"]["j"]),
                    np.array(phi[i]) * (j[i] ** exponents_dict["phi"]["j"]),
                    alpha=1, s=40, linewidth=line_width, facecolors='none',
                    color=colors[i], marker=markers[i],
                    label=r'$j = $'+r'${:.2f}$'.format(j[i]))

    font = font_manager.FontProperties(family='DejaVu Sans', weight='roman',
                                       style='normal', size='large',
                                       stretch='ultra-condensed')

    ax1.legend(bbox_to_anchor=(0.90, 0.99), loc='upper left',
               edgecolor='white', framealpha=0, prop=font, borderaxespad=0.)
    ax1.set_xlabel(r"$r / j^{2/3}$", size='xx-large', fontweight='black')
    ax1.set_ylabel(r'$\phi / j^{1/3}$', size='xx-large')
    # ax1.set_ylim([5,14])
    ax1.grid(False)
    ax1.tick_params(axis='x', labelsize='large')
    ax1.tick_params(axis='y', labelsize='large')
    ax1.spines['bottom'].set_color('black')
    ax1.spines['top'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['right'].set_color('black')

    plt.savefig('tutorial_plots/widom_scaling_renormalized.pdf',
                format='pdf', dpi=1200)
    plt.show()


def plot_laminar_bingham():
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(projection='3d')

    possible_re_tau = np.linspace(start=100, stop=200, num=5)
    possible_he = np.linspace(start=10, stop=100, num=10)

    j = 0
    for re_tau in possible_re_tau:
        He_ = []
        R_ = []
        U_ = []
        for he in possible_he:
            possible_r = np.linspace(start=he / (re_tau**2), stop=1, num=20)
            for r in possible_r:
                u_ = ((re_tau / 4) * (1 - (r ** 2))) - (
                    (1 - r) * (he / re_tau) / 2)
                if u_ > 0:
                    U_.append(u_)
                    R_.append(r)
                    He_.append(he)

        ax.scatter(R_, He_, U_, color=colors[j], marker=markers[j],
                   label=r'$Re_\tau= $'+r'${:.0f}$'.format(re_tau))
        ax.grid(False)

        j += 1

    font = font_manager.FontProperties(family='DejaVu Sans', weight='roman',
                                       style='normal', size='x-large',
                                       stretch='ultra-condensed')

    ax.legend(bbox_to_anchor=(1, 0.2), loc='upper left', edgecolor='white',
              framealpha=0, prop=font, borderaxespad=0.)
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.set_xlabel(r"$r$", size='xx-large', labelpad=10)
    ax.set_ylabel(r"$He$", size='xx-large', labelpad=20)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r"$u^+$", size='xx-large', labelpad=20, rotation=0)
    ax.view_init(elev=5, azim=80)

    plt.savefig('tutorial_plots/laminar_bingham_flow.pdf', format='pdf',
                dpi=1200)
    plt.show()


def plot_laminar_bingham_renormalized(exponents_dict):
    r_exponent = - exponents_dict["r^"]["Re_tau"]
    He_exponent = - exponents_dict["He"]["Re_tau"]
    u_plus_exponent = - exponents_dict["u+"]["Re_tau"]

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(projection='3d')

    possible_re_tau = np.linspace(start=100, stop=200, num=5)
    possible_he = np.linspace(start=10, stop=100, num=10)

    j = 0
    for re_tau in possible_re_tau:
        He_ = []
        R_ = []
        U_ = []
        for he in possible_he:
            possible_r = np.linspace(start=he / (re_tau**2), stop=1, num=20)
            for r in possible_r:
                u_ = ((re_tau / 4) * (1 - (r ** 2))) - (
                    (1 - r) * (he / re_tau) / 2)
                if u_ > 0:
                    U_.append(u_)
                    R_.append(r)
                    He_.append(he)

        ax.scatter(np.array(R_) / (re_tau ** r_exponent),
                   np.array(He_) / (re_tau ** He_exponent),
                   np.array(U_) / (re_tau ** u_plus_exponent),
                   color=colors[j], marker=markers[j],
                   label=r'$Re_\tau= $'+r'${:.0f}$'.format(re_tau))
        ax.grid(False)

        j += 1

    font = font_manager.FontProperties(family='DejaVu Sans', weight='roman',
                                       style='normal', size='x-large',
                                       stretch='ultra-condensed')

    ax.legend(bbox_to_anchor=(1, 0.2), loc='upper left', edgecolor='white',
              framealpha=0, prop=font, borderaxespad=0.)
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.set_xlabel(r"$r / Re_\tau^{- 0.001}$", size='xx-large', labelpad=10)
    ax.set_ylabel(r"$He / Re_\tau^{2.22}$", size='xx-large', labelpad=20)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r"$u^+ / Re_\tau^{0.99}$", size='xx-large', labelpad=20,
                  rotation=0)
    ax.view_init(elev=5, azim=80)

    plt.savefig('tutorial_plots/laminar_bingham_flow_renormalized.pdf',
                format='pdf', dpi=1200)
    plt.show()


def _get_nikuradse_data_for_plotting():

    nikuradse_df = pd.read_excel(
        "../Data/Nikuradse_original_data/Nikuradse.xlsx"
    )

    re_key_list = ["Re_15", "Re_30_6", "Re_60", "Re_126", "Re_252", "Re_507",
                   "Re_1300"]
    friction_key_list = ["100lambda_15", "100lambda_30_6", "100lambda_60",
                         "100lambda_126", "100lambda_252", "100lambda_507",
                         "100lambda_1300"]
    D_r_list = [15, 30.6, 60, 126, 252, 507, 1300]

    length_list = [76, 74, 77, 67, 43, 38, 39]
    r_hat_list = []

    for d_r in D_r_list:
        r_hat_list.append(1 / d_r)

    Re_ = []
    R_hat_ = []
    f_ = []

    for i in range(7):
        Re = list(nikuradse_df[re_key_list[i]].values)
        f = list(nikuradse_df[friction_key_list[i]].values)
        R_hat = r_hat_list[i]

        Re_.append(Re[:length_list[i]])
        f_.append(f[:length_list[i]])
        R_hat_.append(R_hat)

    return Re_, R_hat_, f_


def plot_nikuradse_data():

    Re, R_hat, f = _get_nikuradse_data_for_plotting()

    fig, ax1 = plt.subplots(figsize=(16, 10))

    left, bottom, width, height = [0.57, 0.55, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])

    j = 0
    for r_hat in R_hat:
        ax1.scatter(Re[j], f[j], alpha=1, s=40, linewidth=line_width,
                    facecolors='none', color=colors[j], marker=markers[j],
                    label=r'$D/r = $'+r'${:.1f}$'.format(1 / r_hat))

        ax2.scatter(Re[j], f[j], alpha=1, s=40, linewidth=line_width,
                    facecolors='none', color=colors[j], marker=markers[j])

        j += 1

    font = font_manager.FontProperties(family='DejaVu Sans', weight='roman',
                                       style='normal', size='large',
                                       stretch='ultra-condensed')

    ax1.legend(bbox_to_anchor=(0.3, 0.99), loc='upper left', edgecolor='white',
               framealpha=0, prop=font, borderaxespad=0.)
    ax1.set_xlabel(r"$Re$", size='xx-large', fontweight='black')
    ax1.set_ylabel(r'$f$', size='xx-large')
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
    ax2.set_xlabel(r"$\log \ Re $", fontsize='large')
    ax2.set_ylabel(r'$\log \ f$', fontsize='large')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', labelsize='medium')
    ax2.tick_params(axis='y', labelsize='medium')
    # ax2.set_xlim([0.01,1])
    # ax2.set_ylim([5,14])
    ax2.grid(False)

    plt.savefig('tutorial_plots/nikuradse_data.pdf',
                format='pdf', dpi=1200)
    plt.show()


def plot_nikuradse_data_renormalized():

    Re, R_hat, f = _get_nikuradse_data_for_plotting()

    fig, ax1 = plt.subplots(figsize=(16, 10))

    left, bottom, width, height = [0.57, 0.17, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])

    j = 0
    for r_hat in R_hat:
        re = np.array(Re[j])
        ff = np.array(f[j])

        ax1.scatter((re**(0.7815293)) * r_hat, ff * (re**(0.22047572)),
                    alpha=1, s=40, linewidth=line_width, facecolors='none',
                    color=colors[j], marker=markers[j],
                    label=r'$D/r = $'+r'${:.1f}$'.format(1 / r_hat))

        ax2.scatter((re**(0.7815293)) * r_hat, ff * (re**(0.22047572)),
                    alpha=1, s=40, linewidth=line_width, facecolors='none',
                    color=colors[j], marker=markers[j])

        j += 1

    font = font_manager.FontProperties(family='DejaVu Sans', weight='roman',
                                       style='normal', size='large',
                                       stretch='ultra-condensed')

    ax1.legend(bbox_to_anchor=(0., 0.99), loc='upper left', edgecolor='white',
               framealpha=0, prop=font, borderaxespad=0.)
    ax1.set_xlabel(r"$Re^{0.78} \ \left( r / D \right)$", size='xx-large',
                   fontweight='black')
    ax1.set_ylabel(r'$Re^{0.22} \ f$', size='xx-large')
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
    ax2.set_xlabel(
        r"$\log \ \left( Re^{0.78} \ \left( r / D \right) \right) $",
        fontsize='large')
    ax2.set_ylabel(r'$\log \ \left( Re^{0.22} \ f \right)$', fontsize='large')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', labelsize='medium')
    ax2.tick_params(axis='y', labelsize='medium')
    # ax2.set_xlim([0.01,1])
    # ax2.set_ylim([5,14])
    ax2.grid(False)

    plt.savefig('tutorial_plots/nikuradse_data_renormalized.pdf',
                format='pdf', dpi=1200)
    plt.show()


def plot_nikuradse_data_goldenfeld_exponents():

    Re, R_hat, f = _get_nikuradse_data_for_plotting()

    fig, ax1 = plt.subplots(figsize=(16, 10))

    left, bottom, width, height = [0.57, 0.17, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])

    j = 0
    for r_hat in R_hat:
        re = np.array(Re[j])
        ff = np.array(f[j])

        ax1.scatter((re**(3/4)) * r_hat, ff * (re**(1/4)),
                    alpha=1, s=40, linewidth=line_width, facecolors='none',
                    color=colors[j], marker=markers[j],
                    label=r'$D/r = $'+r'${:.1f}$'.format(1 / r_hat))

        ax2.scatter((re**(3/4)) * r_hat, ff * (re**(1/4)),
                    alpha=1, s=40, linewidth=line_width, facecolors='none',
                    color=colors[j], marker=markers[j])

        j += 1

    font = font_manager.FontProperties(family='DejaVu Sans', weight='roman',
                                       style='normal', size='large',
                                       stretch='ultra-condensed')

    ax1.legend(bbox_to_anchor=(0., 0.99), loc='upper left', edgecolor='white',
               framealpha=0, prop=font, borderaxespad=0.)
    ax1.set_xlabel(r"$Re^{3/4} \ \left( r / D \right)$", size='xx-large',
                   fontweight='black')
    ax1.set_ylabel(r'$Re^{1/4} \ f$', size='xx-large')
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
    ax2.set_xlabel(r"$\log \ \left( Re^{3/4} \ \left( r / D \right) \right)$",
                   fontsize='large')
    ax2.set_ylabel(r'$\log \ \left( Re^{1/4} \ f \right)$', fontsize='large')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', labelsize='medium')
    ax2.tick_params(axis='y', labelsize='medium')
    # ax2.set_xlim([0.01,1])
    # ax2.set_ylim([5,14])
    ax2.grid(False)

    plt.savefig('tutorial_plots/nikuradse_data_goldenfeld_exponents.pdf',
                format='pdf', dpi=1200)
    plt.show()


def plot_HB_data():

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(projection='3d')

    possible_re_tau = np.linspace(start=100, stop=200, num=5)
    possible_he = np.linspace(start=10, stop=100, num=10)
    possible_n = [0.3, 0.5, 1.0]

    i = 0
    j = 0
    for n in possible_n:
        for re_tau in possible_re_tau:
            U_ = []
            R_ = []
            He_ = []
            for he in possible_he:
                phi = he / (re_tau ** 2)
                possible_r = np.linspace(start=phi, stop=1, num=20)
                for r in possible_r:
                    u_ = (re_tau / 4) * (
                        (1 - phi)**((n+1)/n) - (r - phi)**((n+1)/n))
                    if u_ > 0:
                        U_.append(u_)
                        R_.append(r)
                        He_.append(he)

            ax.scatter(R_, He_, U_, color=colors[i], marker=markers[j % 8],
                       label=r'$Re_\tau= $'+r'${:.0f}$'.format(re_tau) + r', $n= $' + r'${:.1f}$'.format(n))
            ax.grid(False)

            j += 1
        i += 1

    font = font_manager.FontProperties(family='DejaVu Sans', weight='roman',
                                       style='normal', size='x-large',
                                       stretch='ultra-condensed')

    ax.legend(bbox_to_anchor=(0.9, 0.8), loc='upper left', edgecolor='white',
              framealpha=0, prop=font, borderaxespad=0.)
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.set_xlabel(r"$r$", size='xx-large', labelpad=10)
    ax.set_ylabel(r"$He$", size='xx-large', labelpad=20)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r"$u^+$", size='xx-large', labelpad=20, rotation=0)
    ax.view_init(elev=5, azim=80)

    plt.savefig('tutorial_plots/laminar_HB_flow.pdf', format='pdf',
                dpi=1200)
    plt.show()


def plot_HB_data_renormalized(exponents_dict):
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(projection='3d')

    possible_re_tau = np.linspace(start=100, stop=200, num=5)
    possible_he = np.linspace(start=10, stop=100, num=10)
    possible_n = [0.3, 0.5, 1.0]

    i = 0
    j = 0
    for n in possible_n:
        r_exponent = exponents_dict["n=" + str(n)]["r^"]["Re_tau"]
        he_exponent = exponents_dict["n=" + str(n)]["He"]["Re_tau"]
        u_exponent = exponents_dict["n=" + str(n)]["u+"]["Re_tau"]
        for re_tau in possible_re_tau:
            U_ = []
            R_ = []
            He_ = []
            for he in possible_he:
                phi = he / (re_tau ** 2)
                possible_r = np.linspace(start=phi, stop=1, num=20)
                for r in possible_r:
                    u_ = (re_tau / 4) * (
                        (1 - phi)**((n+1)/n) - (r - phi)**((n+1)/n))
                    if u_ > 0:
                        U_.append(u_)
                        R_.append(r)
                        He_.append(he)

            ax.scatter(np.array(R_) * (re_tau ** r_exponent),
                       np.array(He_) * (re_tau ** he_exponent),
                       np.array(U_) * (re_tau ** u_exponent), color=colors[i],
                       marker=markers[j % 8],
                       label=r'$Re_\tau= $'+r'${:.0f}$'.format(re_tau) + r', $n= $' + r'${:.1f}$'.format(n))
            ax.grid(False)

            j += 1
        i += 1

    font = font_manager.FontProperties(family='DejaVu Sans', weight='roman',
                                       style='normal', size='x-large',
                                       stretch='ultra-condensed')

    ax.legend(bbox_to_anchor=(0.9, 0.8), loc='upper left', edgecolor='white',
              framealpha=0, prop=font, borderaxespad=0.)
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.set_xlabel(r"$r$", size='xx-large', labelpad=10)
    ax.set_ylabel(r"$He$", size='xx-large', labelpad=20)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r"$u^+$", size='xx-large', labelpad=20, rotation=0)
    ax.view_init(elev=5, azim=80)

    plt.savefig('tutorial_plots/laminar_HB_flow_renormalized.pdf', format='pdf',
                dpi=1200)
    plt.show()
