import matplotlib.pyplot as plt

def plot(x,y):
    # plt.xticks(range(min(x), max(x) + 1, 1000))
    plt.plot(x, y, 'r-.')
    plt.show()