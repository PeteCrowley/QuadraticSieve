from sympy import prime
import random

out_file = "huge_testcases.txt"
range_start = 1000000
range_end = 2000000

tests = []
for i in range(100):
    p1 = prime(random.randint(range_start, range_end))
    # primes must be distinct
    while (p2 := prime(random.randint(range_start, range_end))) == p1:
        pass
    N = p1 * p2
    tests.append((N, p1, p2))

with open(out_file, "w") as f:
    for test in tests:
        f.write(f"{test[0]} {test[1]} {test[2]}\n")
