class Car:
    def __init__(self, name, gas):
        self.name = name
        self.gas = gas

    def move(self):
        if self.gas > 0:
            self.gas = self.gas - 1
            print("{}: move".format(self.name))
        else:
            print("{}: stop".format(self.name))


car1 = Car('kbox', 3)
car2 = Car('Kwagon', 5)

for i in range(5):
    car1.move()
    car2.move()
