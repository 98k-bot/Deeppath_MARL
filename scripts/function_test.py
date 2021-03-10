import math
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager

plt.style.use('seaborn-whitegrid')
font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 14}
plt.rcParams["font.family"] = "Times New Roman"
matplotlib.rc('font', **font)

f = open("test.txt", "r")
div = []
alpha = []
for x in f:
  div.append(float(x))
  alpha.append(math.exp(1-float(x)))
print(div)
print(alpha)
div1 = sorted(div,reverse=False)
alpha1 = sorted(alpha,reverse=True)
plt.plot(div1, alpha1, '-xk')
plt.xlabel(r"$cos(p,p_i)$")
plt.ylabel(r"$\alpha = e^{-cos(p,p_i)+1}$")
plt.savefig('functionTest.pdf', bbox_inches='tight')
plt.show()