import numpy as np
from sympy.ntheory import factorint
from sympy import primerange
import sys
import time
from galois import GF2
import math



def choose_factor_base_bound(N) -> int:
    # A random formula I found online
    L = np.exp(np.sqrt(np.log(N) * np.log(np.log(N))))
    B = L ** (1/2)
    return math.ceil(B)

def build_factor_base(B) -> list:
    return primerange(2, B+1) # returns all primes from 2 to B


# TODO: Actually implement sieve here
def find_b_smooth_squares(N, B: int) -> tuple[list, list]:
    nums = []
    factorizations = []
    i = math.ceil(np.sqrt(N))     # start at sqrt(N)
    
    while len(nums) < B + 1:
        x = pow(i, 2, N)    # find i^2 mod N
        factors = factorint(x)
        # if i^2 is B smooth then add the factorization we found
        if all([p < B for p in factors]):
            nums.append(i)
            factorizations.append(factors)
        i += 1
    # print(nums, factorizations)
    return nums, factorizations


def vectorize_factorizations(factorizations, dict_factor_base) -> np.array:
    # Build the B+1 by B matrix mod 2
    matrix = [[0 for _ in dict_factor_base] for _ in factorizations]
    # for each equation
    for i, factors in enumerate(factorizations):
        # for each prime in the factorization
        for p, exponent in factors.items():
            matrix[i][dict_factor_base[p]] = exponent % 2 # set it to the exponent mod 2
    return matrix

def quadratic_sieve(N):
    B = choose_factor_base_bound(N)
    factor_base = list(build_factor_base(B))
    dict_factor_base = {p: i for i, p in enumerate(factor_base)}
    nums, factorizations = find_b_smooth_squares(N, B)
    vectorized_factorizations = vectorize_factorizations(factorizations, dict_factor_base)
    # Find nontrival solution of Ax = 0 mod 2
    A = GF2(np.matrix(vectorized_factorizations).T)
    # find the null space of A
    null_space = A.null_space()
    
    # any row vector will work as a linear combo that yields 0 mod 2
    for row in null_space:
        linear_combo = row
        # each 1 in this row represents a congruence we should include
        exponents_squared = [0 for _ in factor_base]
        sqrt_number = 1
        for i, included in enumerate(linear_combo):
            if included == 1:
                sqrt_number = (nums[i] * sqrt_number) % N   # we actually returned the sqrt of x from b_smooth method so this is fine
                # add the exponents of the factorization to our list we will take the product of
                for p, exponent in factorizations[i].items():
                    exponents_squared[dict_factor_base[p]] += exponent
                nums.append(nums[i])

        exponents = [e // 2 for e in exponents_squared] # we want the square root of the product of the p^e's
        # so we just divide all the exponents by 2 and take the product
        exp_product = 1
        for p, exponent in dict_factor_base.items():
            exp_product = (exp_product * pow(p, exponents[exponent], N)) % N

        # works if x != +- y mod n, otherwise we keep going and try a different linear combo
        if exp_product != sqrt_number and exp_product != -sqrt_number % N:
            # then a common factor is gcd(sqrt_number - exp_product, N)
            one_factor = math.gcd(sqrt_number - exp_product, N)
            other_factor = N // one_factor      # and another is just dividing N by the first
            return one_factor, other_factor
    
    # if we don't find a solution, return None
    return None

if __name__ == '__main__':
    # Number to factor
    N = int(sys.argv[1])
    time_start = time.time()
    print(quadratic_sieve(N))
    time_end = time.time()
    print("Time elapsed: ", round(time_end - time_start, 2), "seconds")
