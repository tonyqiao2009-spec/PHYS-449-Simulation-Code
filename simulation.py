import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline
from scipy.integrate import quad
import pandas as pd

hbar_eV = 6.582119569*10**(-16)



class TempSim:

    def __init__(self, l_temp, e_temp):
        self.lattice_temp = l_temp
        self.electron_temp = e_temp
        self.k_b = 1.38*10**(-23)
        self.hbar = 6.63*10**(-34)/(2*np.pi)
        self.e_heatcapacity = 4.075774971297359*10**(-3)
        self.spec_func = self.get_spectral_func()
        self.lattice_heat_capacity = self.get_lattice_heat_capacity()

    def get_spectral_func(self):
        file_path_spectral = 'Excel Sheets/Spectral Function.csv'
        data = np.loadtxt(file_path_spectral, delimiter=',')
        x = data[:,0]*10**(-3)/hbar_eV
        y = data[:,1]
        spec_func = make_smoothing_spline(x, y, lam=0)
        return spec_func
    
    def get_lattice_heat_capacity(self):
        file_path_lattice_heat_capacity = 'Excel Sheets/Lattice Heat Capacity.csv'
        data = np.loadtxt(file_path_lattice_heat_capacity, delimiter=',')
        x = data[:,0]
        y = data[:,1]
        lattice_heat_capacity = make_smoothing_spline(x, y, lam=0)
        return lattice_heat_capacity

    def set_lattice_temp(self, l_temp):
        self.lattice_temp = l_temp

    def set_electron_temp(self, e_temp):
        self.electron_temp = e_temp

    def bose_l(self, w):
        beta = 1/(self.k_b * self.lattice_temp)
        energy = self.hbar*w
        return np.reciprocal(np.exp(beta*energy)-1)
    
    def bose_e(self, w):
        beta = 1/(self.k_b * self.electron_temp)
        energy = self.hbar*w
        return np.reciprocal(np.exp(beta*energy)-1)
    
    def g(self):
        constant = (6*(self.hbar)**(2)*self.e_heatcapacity)/(np.pi*(self.k_b)**(2))
        integrand = lambda w: w**2*self.spec_func(w)*(self.bose_l(w)-self.bose_e(w))
        min = int(10**(-3)/hbar_eV)
        max = int(100*10**(-3)/hbar_eV)

        
        return constant*quad(integrand, 1.1*min, max)[0]
    
    def gaussian(self, time, sigma=50*10**(-15), Amplitude = 57*10**(-6)):
        return Amplitude*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-1*(time)**2/(2*sigma**2))




# set starting equilibrium temperature
T_initial = 20
# set initial time
t = -0.1*10**(-12)
Te = T_initial
Tl = T_initial

ts = [t]
Tes = [Te]
Tls = [Tl]

# set number of steps
n = 1000
# set final time
t_f = 0.5*10**(-12)

dt = (t_f-t)/n

model = TempSim(Te, Tl)
for i in range(n):
    g = model.g()
    P = model.gaussian(t)
    C_e = model.e_heatcapacity*Te

    Tl = Tl-(1*g/(model.lattice_heat_capacity(Tl)))*dt
    Te = Te+(g/C_e+P/C_e)*dt
  
    model.set_electron_temp(Te)
    model.set_lattice_temp(Tl)

    t = t+dt
    ts.append(t)
    Tes.append(Te)
    Tls.append(Tl)


file_path_save = 'Temperature Series/'
series = np.stack([ts, Tes, Tls], axis=1)
df = pd.DataFrame(series, columns=['t', 'Te', 'Tl'])
# save folder
df.to_csv(file_path_save+str(T_initial)+'.csv', index=False)
