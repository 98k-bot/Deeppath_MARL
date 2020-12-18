import multiprocessing
from multiprocessing import Pool
import random

output=[]
data = range(0,4)
data2 = 0
def f(x,y,z):
    return x,y,z

def handler():
    p = multiprocessing.Pool(processes=4)
    r=p.map(f, data,data2)
    r = p.starmap(f, [(data, 1, 'a')]*3)
    return r


if __name__ == '__main__':
    output.append(handler())

print(output[0])