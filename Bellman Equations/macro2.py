import numpy as  np
import matplotlib.pyplot as plt
def v(a,b):
    return np.log(np.maximum(a**0.3-b+0.2*a,1e-8))
k1 = np.linspace(0.04,5,1000)                 #discrete capital space
k2 = np.zeros(len(k1))                  #guess policy function
k3 = np.zeros(len(k1))                                #guess value function
k_value = np.zeros(len(k1))                  #updated value function
k_policy = np.zeros(len(k1))                 #updated policy function

z = 0
while z<100:                                             #iterating through value function
    for i in range(len(k1)):                                           #iterating through policy func by setting one k and finding k' that maximises the function
        a = k1[i]
        max_value = -np.inf
        max_index = 0
        for j in range(len(k1)):
            current_value = v(a,k1[j])+0.6*k3[j]
            if current_value>max_value:
                max_value = current_value
                max_index = j
        k_policy[i] = k1[max_index]
        k_value[i] = max_value

    if np.linalg.norm(k3-k_value)<1e-6 and np.linalg.norm(k2-k_policy)<1e-6:
        break
    k2 = k_policy.copy()
    k3 = k_value.copy()
    if z%3==0 and z!=0:
        #plt.plot(k1,k_value,label = f'iteration{z}')
        #plt.xlabel("Level of capital K")
        #plt.ylabel("Value function w(k)")
        #plt.title("Value Function iteration")
        plt.plot(k1,k_policy,label = f'iteration{z}')
        plt.xlabel("Level of capital K")
        plt.ylabel("Policy function w(k)")
        plt.title("Policy Function iteration")
    z+=1
print(f"converged in {z} iterations")
plt.legend()
plt.show()
for i in range(len(k_policy)):
    if k_policy[i] == k1[i]:
        print(k1[i])
#converged in 36 iterations
#0.10355835583558357
#0.10365436543654366