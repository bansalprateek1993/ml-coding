import numpy as np

def moving_average(x, k):
    if len(x) == 0:
        return np.array([])

    if k <= 0:
        raise ValueError("K must be positive")

    if k > len(x):
        raise ValueError("k cannot be larger than input length")

    result = []
    for i in range(len(x)-k+1):
        window = x[i:i+k]
        result.append(np.mean(window))

    return np.array(result)

def optimized_ma(x, k):
    if len(x) == 0:
        return np.array([])

    if k > len(x):
        raise ValueError("k cant be more then x")

    if k < 0:
        raise ValueError("k cannot be less than 0")

    ans = []
    weighted_sum = np.sum(x[:k])
    ans.append(weighted_sum/k)

    for i in range(k, len(x)):
        weighted_sum += x[i]
        weighted_sum -= x[i-k]
        ans.append(weighted_sum/k)
    return ans

if __name__ == "__main__":
    x = np.array([1,2,3,4,5])
    k = 3
    w = np.array([0.2, 0.3, 0.5])

    print(moving_average(x, k))
    # print(weighted_moving_average(x, w, k))
    print(optimized_ma(x, k))
    # print(optimized_weighted_ma(x, w, k))