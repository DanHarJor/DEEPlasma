from tools import get_test_points
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
    _, test_dir = sys.argv
    x,y = get_test_points(test_dir)
    fig = plt.figure()
    plt.hist(x[:,0])
    fig.savefig('./no_boundary_test.png')
    
    