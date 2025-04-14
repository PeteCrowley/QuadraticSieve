import numpy as np
from sympy.ntheory import factorint
from sympy import primerange, sqrt_mod
import sys
import time
from galois import GF2
import math



def choose_factor_base_bound(N) -> int:
    """Chooses the bound for the factor base using the formula from the book"""
    L = math.exp(math.sqrt(math.log(N) * math.log(math.log(N))))
    B = L ** (1/2)
    return math.ceil(B)

def is_quadratic_residue(N, p):
    """Returns True if N is a square mod p, False otherwise.
       
        :param N: The number to check
        :param p: The prime to check against
        :returns boolean: True if N is a square mod p, False otherwise
    """
    if pow(N, (p - 1) // 2, p) == 1:
        return True
    return False

def b_smooth_factor(x: int, factor_base: list) -> dict | None:
    """Trial division to check if x is B-Smooth and if so to find factorization

        :param x: The number to factor
        :param factor_base: The factor_base
        :returns powers: A dictionary of the form {p: exponent} if x is B-smooth, None otherwise
    """
    powers = dict()
    for p in factor_base:
        while x % p == 0:
            if p not in powers:
                powers[p] = 1
            else:
                powers[p] += 1
            x //= p
            if x == 1:
                return powers
    if x != 1:
        return None
    return powers

def build_smart_factor_base(N, B) -> list:
    """Builds the factor base of primes less than B that are quadratic residues mod N

        :param N: The number to factor
        :param B: The bound for the factor base
        :returns factor_base: A list of primes that are quadratic residues mod N
    """
    factor_base = []
    for p in primerange(2, B+1):
        if is_quadratic_residue(N, p):
            factor_base.append(p)
    return factor_base

def better_find_b_smooth_squares(N, B, factor_base, dict_factor_base) -> list:
    """Finds B-smooth squares mod N using the quadratic sieve method

        :param N: The number to factor
        :param B: The bound for the factor base
        :param factor_base: The factor base
        :param dict_factor_base: A dictionary of the form {p: index} for the factor base
        :returns nums: A list of numbers that are B-smooth squares mod N
        :returns factorizations: A list of dictionaries of the form {p: exponent} for each number in nums
    """
    def f(x):
        return x**2 - N
    
    log_factor_base  = np.log(factor_base)

    sieve_interval = 100_000
    sieve_start = math.ceil(math.sqrt(N))

    nums = []
    factorizations = []

    # we want enough numbers to guarantee a linear dependence
    while len(nums) < len(factor_base) + 5:
        registers = [0] * (sieve_interval)     # check from s to s + A, initialize to log of the number


        for p in factor_base:
            sols = sqrt_mod(N, p, all_roots=True)   # find the two solutions (if they exist)
            for sol in sols:
                for i in range((sol-sieve_start) % p, sieve_interval, p):   # all numbers congruent to sol mod p are also solutions
                    registers[i] += log_factor_base[dict_factor_base[p]]    # so we add the log of the prime to the register

        

        # for numbers with small values now, we will check if they are B smooth
        for x in range(0, sieve_interval):
            if registers[x] >= math.log(x+sieve_start) - math.log(B):
                factors = b_smooth_factor(f(x+sieve_start), factor_base)
                if factors is not None:
                    # if the number is B smooth, we will add it to our list
                    nums.append(x+sieve_start)
                    factorizations.append(factors)
                    # print(len(nums), nums[-1], factorizations[-1])
                    if len(nums) >= len(factor_base) + 1:
                        break

        sieve_start += sieve_interval
        if sieve_start > N:
            break
        
    return nums, factorizations


def vectorize_factorizations(factorizations, dict_factor_base) -> np.array:
    """Vectorizes the factorizations into a matrix mod 2

        :param factorizations: A list of dictionaries of the form {p: exponent} for each number in nums
        :param dict_factor_base: A dictionary of the form {p: index} for the factor base
        :returns matrix: A matrix of size (len(factorizations), len(dict_factor_base)) mod 2
    """
    # Build the B+1 by B matrix mod 2
    matrix = [[0 for _ in dict_factor_base] for _ in factorizations]
    # for each equation
    for i, factors in enumerate(factorizations):
        # for each prime in the factorization
        for p, exponent in factors.items():
            matrix[i][dict_factor_base[p]] = exponent % 2 # set it to the exponent mod 2
    return matrix

def quadratic_sieve(N, verbose=False):
    """Quadratic Sieve implementation to factor N

        :param N: The number to factor
        :param verbose: If True, print debug information
        :returns factors: A tuple of the two factors if found, None if it can't find any"""
    B = choose_factor_base_bound(N)
    factor_base = list(build_smart_factor_base(N, B))
    # print(len(factor_base), factor_base)
    if verbose:
        print(f"=== Chose factor base of {B} consisting of {len(factor_base)} primes===")
    dict_factor_base = {p: i for i, p in enumerate(factor_base)}

    if verbose:
        print("=== Searching for B-smooth squares ===")
    nums, factorizations = better_find_b_smooth_squares(N, B, factor_base, dict_factor_base)

    if nums == []:
        return None
    
    vectorized_factorizations = vectorize_factorizations(factorizations, dict_factor_base)
    # Find nontrival solution of Ax = 0 mod 2
    if verbose:
        print("=== Finding linear dependence===")
    A = GF2(np.matrix(vectorized_factorizations).T)
    # find the null space of A
    null_space = A.null_space()
    
    if verbose:
        print("=== Trying to factor n ===")
    # any row vector will work as a linear combo that yields 0 mod 2
    for row in null_space:
        linear_combo = row
        # print(row)
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
            if verbose:
                print("=== Found factors ===")
                print(f"{N} = {one_factor} * {other_factor}")
            return one_factor, other_factor
    
    # if we don't find a solution, return None
    return None

if __name__ == '__main__':
    # Number to factor
    N = int(sys.argv[1])
    print(N)
    time_start = time.time()
    quadratic_sieve(N, verbose=True)
    time_end = time.time()
    print("Time elapsed: ", round(time_end - time_start, 2), "seconds")
