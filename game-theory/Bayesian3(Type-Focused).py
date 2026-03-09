import math
import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
def bern(p):
    if np.random.rand()>p:
        return 0
    else:
        return 1
def random_joint_distribution(n):
    probs = np.random.rand(n)  # random numbers from uniform[0,1)
    return probs / probs.sum()

def bayesian_climate_game(a,n,l):
    b = 0

    strategy_matrix = np.zeros((a, n))
    while b<a:
       param = np.random.rand(6)  # beta, alpha, gamma, pl,ph,S
       param[4], param[3] = max(param[3] / (param[3] + param[4]), param[4] / (param[3] + param[4])), min(param[3] / (param[3] + param[4]), param[4] / (param[3] + param[4]))
       param[2] = max(param[3] / param[4], param[2])
       type_matrix = np.zeros((3, n))
       for i in range(type_matrix.shape[0]):               #Initialising the type matrix 0 = energy type, 1 = political type, 2 = climate type
           for j in range(type_matrix.shape[1]):
                type_matrix[i,j]= bern(0.5)
       m = 0                                               #Let it be a partial pooling equlibrium where high political types play lead whereas others play veto
       while m<n:
           lead = 0
           swing = 0
           veto = 0
           k = 0
           bel = random_joint_distribution(l)
           for c in range(l):
               #approx[c,:] = sps.multinomial.rvs(n,bel)
               #if approx[c,:4].sum()>=n*(1/2):             #This line assumes that first 4 types are enough to force a coalition
               if bern(bel[c]) == 1 :                         #probability of coalition formation(there always exists a swing state by assumption)
                  lead += (-type_matrix[0,m]*(1/param[0])+type_matrix[1,m]-param[3]*type_matrix[2,m]-param[5])
                  swing +=  (
                              -type_matrix[0, m] + param[1]*type_matrix[1, m] - (param[3] * type_matrix[2, m]+param[5]))
                  veto += (
                          (-type_matrix[0, m] * (param[0]) )+ (-type_matrix[1, m]*(1/param[1])) - (param[3]*(1/param[2]) * type_matrix[2, m]))
               else:
                   lead += (
                           (param[0]*type_matrix[0, m])+param[1]*type_matrix[1, m] - (param[4] *param[2]* type_matrix[2, m]))
                   swing += (
                           type_matrix[0, m] - param[1] * type_matrix[1, m] - (
                               param[4] * type_matrix[2, m]))
                   veto += (
                           type_matrix[0, m]  + type_matrix[1, m] - (
                               param[4] * type_matrix[2, m]))
               #bel = [approx[c,i]/approx[c,:].sum() for i in range(7)]
           ap_lead = lead/l
           ap_swing = swing/l
           ap_veto = veto/l
           if ap_lead>= ap_swing and ap_lead>= ap_veto:
               strategy_matrix[b,m] = 2
           elif ap_swing>= ap_lead and ap_swing>=ap_veto:
               strategy_matrix[b, m] = 1
           elif ap_veto>= ap_lead and ap_veto>= ap_swing:
               strategy_matrix[b, m] = 0
           m+=1
       b+=1
    return strategy_matrix,type_matrix,param
blegh,blegh2,blegh3 = bayesian_climate_game(100,500,1000000)


def per_type_strategy_graphs(strategy_matrix, type_matrix, param_values):
    """
    Generates separate bar plots for each type, showing strategy counts with readable labels.

    Parameters:
    strategy_matrix : (a x n) array
    type_matrix : (3 x n) array
    param_values : array of parameter values used in simulation
    """

    assoc_dict = {i: {'lead': 0, 'swing': 0, 'veto': 0} for i in range(8)}
    n_agents = type_matrix.shape[1]

    # Compute type codes 0-7
    type_codes = (type_matrix[0, :].astype(int) << 2) + (type_matrix[1, :].astype(int) << 1) + (
        type_matrix[2, :].astype(int))

    # Count strategy choices by type
    for run in strategy_matrix:
        for idx in range(n_agents):
            tcode = int(type_codes[idx])
            strat = run[idx]
            if strat == 2:
                assoc_dict[tcode]['lead'] += 1
            elif strat == 1:
                assoc_dict[tcode]['swing'] += 1
            elif strat == 0:
                assoc_dict[tcode]['veto'] += 1

    # Plot separate graph per type if that type exists
    for i in range(8):
        energy = (i >> 2) & 1
        political = (i >> 1) & 1
        climate = i & 1

        total = assoc_dict[i]['lead'] + assoc_dict[i]['swing'] + assoc_dict[i]['veto']

        if total == 0:
            continue  # Skip types not realised

        labels = ['Veto (0)', 'Swing (1)', 'Lead (2)']
        counts = [assoc_dict[i]['veto'], assoc_dict[i]['swing'], assoc_dict[i]['lead']]
        colors = ['red', 'orange', 'green']

        plt.figure(figsize=(5, 4))
        plt.bar(labels, counts, color=colors)

        title_str = f"θₑ={energy}, θₚ={political},θ₍c₎ ={climate}"
        plt.title(f"Strategy Counts for Type: {title_str}")
        plt.ylabel('Total Count Across Runs')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()

    # Print parameter values
    print("Parameter Values Used:")
    print(
        f"β = {param_values[0]:.3f}, α = {param_values[1]:.3f}, γ = {param_values[2]:.3f}, pₗ = {param_values[3]:.3f}, pₕ = {param_values[4]:.3f},s = {param_values[5]:.3f}")
per_type_strategy_graphs(blegh, blegh2, blegh3)
def count_coalitions(strategy_matrix):
    """
    Counts the number of coalition-forming rounds.
    Coalition forms if n_Lead + n_Swing >= n_v
    """
    coalition_formed = []
    for row in strategy_matrix:
        n_L = np.sum(row == 2)
        n_S = np.sum(row == 1)
        n_v = np.sum(row == 0)
        coalition_formed.append(1 if (n_L + n_S) >= n_v else 0)
    return coalition_formed

# 2. Plot coalition formation over simulation rounds
def plot_coalition_formation(coalition_list):
    plt.figure(figsize=(8, 3))
    plt.plot(coalition_list, drawstyle='steps-mid', label="Coalition Formed", color='blue')
    plt.xlabel("Round")
    plt.ylabel("Coalition Status")
    plt.title("Coalition Formation Over Rounds")
    plt.yticks([0, 1], labels=["No", "Yes"])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
coalition_list = count_coalitions(blegh)
plot_coalition_formation(coalition_list)
def type_strategy_association_with_graphs(strategy_matrix, type_matrix):
    """
    Associates type realisations with chosen strategies and plots results.

    Parameters:
    strategy_matrix : np.array (a x n)
        Strategy choices for each agent in each run.
    type_matrix : np.array (3 x n)
        Type realisations for each agent (energy, political, climate types).
    """
    assoc_dict = {i: {'lead': 0, 'swing': 0, 'veto': 0} for i in range(8)}

    n_agents = type_matrix.shape[1]

    # Convert types to unique integers 0-7
    type_codes = (type_matrix[0, :].astype(int) << 2) + (type_matrix[1, :].astype(int) << 1) + (
        type_matrix[2, :].astype(int))

    # Count strategy choices for each type
    for run in strategy_matrix:
        for idx in range(n_agents):
            tcode = int(type_codes[idx])
            strat = run[idx]
            if strat == 2:
                assoc_dict[tcode]['lead'] += 1
            elif strat == 1:
                assoc_dict[tcode]['swing'] += 1
            elif strat == 0:
                assoc_dict[tcode]['veto'] += 1

    # Prepare data for plotting
    types = [f"{i:03b}" for i in range(8)]
    leads = [assoc_dict[i]['lead'] for i in range(8)]
    swings = [assoc_dict[i]['swing'] for i in range(8)]
    vetos = [assoc_dict[i]['veto'] for i in range(8)]

    x = np.arange(8)
    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(x - width, vetos, width=width, color='red', label='Veto (0)')
    plt.bar(x, swings, width=width, color='orange', label='Swing (1)')
    plt.bar(x + width, leads, width=width, color='green', label='Lead (2)')

    plt.xticks(x, types)
    plt.xlabel('Type Realisation (Energy, Political, Climate)')
    plt.ylabel('Total Count Across All Runs')
    plt.title('Strategy Choices by Type')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

    return assoc_dict



type_strategy_association_with_graphs(blegh, blegh2)