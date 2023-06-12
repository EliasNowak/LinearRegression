import matplotlib.pyplot as plt
import numpy as np

# model: f(x) = wx + b
# parameters: w,b
# cost function: J(w,b)=1/2m * sum(f(x)-y)^2
# goal: minimize J(w,b)
def init():
    # Dataset
    x_train = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
    y_train = np.array([94, 81, 87, 80, 101, 60, 153, 87, 94, 78, 77, 85, 86])
    # Label chart
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}
    plt.title("Training dataset", loc="left", fontdict=font2)
    plt.xlabel("x, feature")
    plt.ylabel("y, target")
    plt.scatter(x_train, y_train, label="dataset")
    return x_train, y_train
def model(x, w, b):
    return w * x + b
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = model(w, x[i], b)
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost
if __name__ == "__main__":
    x_train, y_train = init()  # init chart

    w = 10
    b = 10
    a = 0.01  # learning rate
    m = x_train.shape[0]

    lowest_cost = compute_cost(x_train, y_train, w, b)

    for _ in range(10000):
        total_sum_w = 0
        total_sum_b = 0
        _sum_w = 0
        _sum_b = 0
        for i in range(m):
            f_wb = model(w, x_train[i], b)
            _sum_w = (f_wb - y_train[i]) * x_train[i]
            _sum_b = (f_wb - y_train[i])
            total_sum_w += _sum_w
            total_sum_b += _sum_b
        total_diff_w = (1 / (2 * m)) * total_sum_w
        total_diff_b = (1 / (2 * m)) * total_sum_b
        w = w - a * total_diff_w
        b = b - a * total_diff_b
        if (lowest_cost > compute_cost(x_train, y_train, w, b)):
            lowest_cost = compute_cost(x_train, y_train, w, b)
            print(lowest_cost)

    y_prediction = []
    for i in range(m):
        y_prediction.append(model(x_train[i], w, b))

    plt.plot(x_train, y_prediction, label="prediction")

    print(f"local minimum: {w} and {b}")
    plt.legend(loc="upper right")
    plt.show()
