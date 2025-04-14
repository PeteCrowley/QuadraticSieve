import os
from quadratic_sieve import quadratic_sieve

def read_test_file(test_file):
    for line in open(test_file):
        N, p1, p2 = map(int, line.strip().split())
        yield N, p1, p2


def test_program(test_file):
    tests = read_test_file(test_file)
    for N, p1, p2 in tests:
        print(f"Testing N={N}, p1={p1}, p2={p2}")
        if (ans := quadratic_sieve(N)) not in [(p1, p2), (p2, p1)]:
            print(f"Failed for N={N}, should be {p1} and {p2} but got {ans}")
            # return
    print("All tests passed!")

if __name__ == '__main__':
    test_program("large_testcases.txt")
    