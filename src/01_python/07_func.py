def sum(start, end):
    result = 0
    for i in range(start, end + 1):
        result = result + i
    return result

print(sum(1, 1))
print(sum(1, 5))
print(sum(1, 10))
