def neuron(x1, x2):
    w1 = 0.8
    w2 = -0.2
    y = w1 * x1 + w2 * x2
    if y > 0.5 :
        return 1
    else :
        return 0

print(neuron(1, 0))
print(neuron(0, 1))
