import numpy as np
import matplotlib.pyplot as plt
import random

def p(r1, r2, alpha):
    r12 = np.sqrt((r1[0]-r2[0])**2+(r1[1]-r2[1])**2+(r1[2]-r2[2])**2)
    result = np.exp(-2*np.linalg.norm(r1))*np.exp(-2*np.linalg.norm(r2))*np.exp(r12/(2*(1+alpha*r12)))

    return result

def localEnergy(r1, r2, alpha):
    r1= np.array(r1)
    r2= np.array(r2)
    r12 = np.sqrt((r1[0]-r2[0])**2+(r1[1]-r2[1])**2+(r1[2]-r2[2])**2)
    r1_unit = r1/np.linalg.norm(r1) if np.linalg.norm(r1) > 1e-12 else np.zeros(3)
    r2_unit = r2/np.linalg.norm(r2) if np.linalg.norm(r2) > 1e-12 else np.zeros(3)

    result = -4+np.dot(r1_unit-r2_unit, r1-r2)*1/(r12*(1+alpha*r12))-1/(r12*(1+alpha*r12)**3)-1/(4*(1+alpha*r12)**4)+1/r12
    return result

def distribution(p,r1,r2,alpha):
    result = p(r1,r2,alpha)**2
    return result

def Metro(func,N,r1,r2, alpha):
    r1i=[]
    r2i=[]
    Ei = []
    distribution_list = []
    r1i.append(r1)
    r2i.append(r2)
    Ei.append(localEnergy(r1,r2,alpha))
    distribution_list.append(distribution(func,r1i[0],r2i[0],alpha))
    step = [-1,1]
    print(func(r1,r2,alpha))
    for i in range(1,N):
        r1trial = [r1i[i-1][0] + random.random()*step[random.randint(0,1)],r1i[i-1][1] + random.random()*step[random.randint(0,1)],r1i[i-1][2] + random.random()*step[random.randint(0,1)]]
        r2trial = [r2i[i-1][0] + random.random()*step[random.randint(0,1)],r2i[i-1][1] + random.random()*step[random.randint(0,1)],r2i[i-1][2] + random.random()*step[random.randint(0,1)]]
        #xtrial= xi[i-1] + random.random()*step[random.randint(0,1)]
        w = distribution(func,r1trial,r2trial,alpha)/distribution(func,r1i[i-1],r2i[i-1],alpha)
        
        if w >= 1:
            r1i.append(r1trial)
            r2i.append(r2trial)
            Ei.append(localEnergy(r1trial,r2trial,alpha))
            distribution_list.append(distribution(func,r1i[i],r2i[i],alpha))
            
        else:
            r = random.random()
            
            if r <= w:
                r1i.append(r1trial)
                r2i.append(r2trial)
                Ei.append(localEnergy(r1trial,r2trial,alpha))
                distribution_list.append(distribution(func,r1i[i],r2i[i],alpha))
                
            else:
                r1i.append(r1i[i-1])
                r2i.append(r2i[i-1])
                Ei.append(localEnergy(r1i[i-1],r2i[i-1],alpha))
                distribution_list.append(distribution(func,r1i[i-1],r2i[i-1],alpha))
                


    return r1i,r2i, Ei, distribution_list

alpha = 0.1


alpha = 0.1
alpha_list = []
Energy_total_list = []
E_var_list = []
while alpha <= 1.0:
    r1,r2,E_local,Distrution = Metro(p,10000,[0.0,0.0,0.0],[1.0,1.0,1.0], alpha)
    E_var = np.var(E_local)
    E_var_list.append(E_var)
    Energy_total = 0.0
    for i in range(len(E_local)):
            Energy_total += E_local[i]

    print("Energy: ", Energy_total/len(E_local))
    Energy_total_list.append(Energy_total/len(E_local))

    alpha_list.append(alpha)
    alpha += 0.02

# Criar subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# Primeiro subplot: Energy vs Alpha
axs[0].plot(alpha_list, Energy_total_list, label='Energy')

axs[0].set_xlabel('Alpha')
axs[0].set_ylabel('Energy')
axs[0].set_title('Energy vs Alpha')
axs[0].legend()
axs[0].grid()

# Segundo subplot: Energy Variance vs Alpha
axs[1].plot(alpha_list, E_var_list, label='Energy Variance')

axs[1].set_xlabel('Alpha')
axs[1].set_ylabel('Energy Variance')
axs[1].set_title('Energy Variance vs Alpha')
axs[1].legend()
axs[1].grid()

# Ajustar layout
plt.tight_layout()
plt.show()
