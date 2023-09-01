import random
import numpy
import time
import torch

def generate_request():
    time_interval = numpy.random.poisson(25, 10)
    return time_interval


def main():
    for i in range(10):
        print(generate_request())



if __name__=="__main__":
    main()
