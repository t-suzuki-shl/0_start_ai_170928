def neuron(x1, x2):
    w1 = 0.8
    w2 = -0.2
    b = -0.5
    y = w1 * x1 + w2 * x2 + b
    if y > 0 :
        return 1
    else :
        return 0

print(neuron(1, 0))
print(neuron(0, 1))
