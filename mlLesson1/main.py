import seaborn

import matplotlib.pyplot as plt
import numpy as np
import torch
# образец 1
def main():
    #from urllib.request import urlopen
    #f = urlopen('https://stepic.org/media/attachments/lesson/16462/boston_houses.csv')
    #sbux = np.loadtxt(f,  skiprows=1, delimiter=",").mean(axis=0)
    

    w = torch.tensor( [[5.,10.],[1.,2.]], requires_grad=True)

    alpha = 0.001

    for _ in range(500):
        # critical: calculate the function inside the loop
        function = (w + 7).log().log().prod()
        function.backward()
        w.data -=  alpha * w.grad
        w.grad.zero_()
        # put code here
        # something is missing

main()


