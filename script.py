import numpy as np

# Bu örneğin basitliği açısından bias'ı sıfır seçtim. Weight ise 2.
# f = w * x
# w = 2
# f = 2 * x

x = np.array((1,2,3,4,10), dtype=np.float32)
y = np.array((2,4,6,8,20), dtype=np.float32)

# İlk weight rastgele seçilir, bu örnekte sıfır seçtim.
w = 0.0

# model prediction
def predict(x):
    return w * x

# calculate error (mean squared error)
def error(y, y_pred):
    return ((y_pred - y)**2).mean()

# w ve b parametrelerinin hangi oranda değiştirileceğinde etkilidir. Bkz: gradient descend.
# MSE = 1/N * (w*x-y)**2
# dJ/dw = 1/N 2x (w*x-y)
def gradient(y, y_pred):
    return np.dot(y, y_pred-y) / y.size


print(f"Prediction before training f(5)={predict(5):.3f}")

learning_rate = 0.01
n_iterations = 40

for epoch in range(n_iterations):
    # prediction = forward pass
    y_head = predict(x)

    # error calculated using mse
    mse = error(y, y_head)

    # gradient
    dw = gradient(y, y_head)

    # update weights
    w -= learning_rate * dw

    if (epoch+1) % 5 == 0:
        print(f"epoch:{epoch+1} w={w:.5f} error={mse:.8f}")


print(f"Prediction after training f(5)={predict(5):.3f}")
