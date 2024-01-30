import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint


def seir_model(y, t, N, beta, sigma, gamma):
    """
    SIER模型的微分方程
    :param y:
    :param t:
    :param N: 总人数
    :param beta: 接触效率
    :param sigma: 感染效率
    :param gamma: 康复效率
    :return:
    """
    S, E, I, R = y
    dSdt = -beta * S * I / N  # 易感者
    dEdt = beta * S * I / N - sigma * E  # 暴露者
    dIdt = sigma * E - gamma * I  # 感染者
    dRdt = gamma * I  # 康复者
    return dSdt, dEdt, dIdt, dRdt


def fit_odeint(t, beta, gamma, sigma):
    return odeint(seir_model, (S0, E0, I0, R0), t, args=(N, beta, gamma, sigma)).T[1]


# Total population, N.
N = 1000
# Initial number of infected, exposed and recovered individuals.
I0, E0, R0 = 1, 0, 0  # 感染者，暴露者，康复者
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - E0 - R0  # 易感人群
# Contact rate, beta; mean incubation rate, sigma; and mean recovery rate, gamma (in 1/days).
beta, sigma, gamma = 1., 1./14, 1./10
# A grid of time points (in days)
T = 500
t = np.linspace(0, T, T)

# Initial conditions vector
y0 = S0, E0, I0, R0
# Integrate the SIER equations over the time grid, t.
ret = odeint(seir_model, y0, t, args=(N, beta, sigma, gamma))
S, E, I, R = ret.T

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
plt.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
plt.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
plt.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of People')
plt.legend()
plt.title('SEIR Model')
plt.show()
