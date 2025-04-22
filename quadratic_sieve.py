import numpy as np
from sympy.ntheory import factorint
from sympy import primerange, sqrt_mod
import sys
import time
from galois import GF2
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


EXTRA_VECTORS = 10
SIEVE_INTERVAL_SIZE = 100_000
PRIMES_TO_SKIP = 5
SIEVE_CUSHION = 20
PARALLEL = True
UPDATE_INTERVAL = 100
BIGGER_B_FACTOR = 4/3


def choose_factor_base_bound(N: int) -> int:
    """Chooses the bound for the factor base using the formula from the book"""
    L = math.exp(math.sqrt(math.log(N) * math.log(math.log(N))))
    B = L ** (1/2)
    B *= BIGGER_B_FACTOR
    return math.ceil(B)

def is_quadratic_residue(N: int, p: int):
    """Returns True if N is a square mod p, False otherwise.
       
        :param N: The number to check
        :param p: The prime to check against
        :returns boolean: True if N is a square mod p, False otherwise
    """
    if pow(N, (p - 1) // 2, p) == 1:
        return True
    return False


def build_smart_factor_base(N: int, B: int) -> list:
    """Builds the factor base of primes less than B that are quadratic residues mod N

        :param N: The number to factor
        :param B: The bound for the factor base
        :returns factor_base: A list of primes that are quadratic residues mod N
    """
    factor_base = [-1]
    for p in primerange(2, B+1):
        if is_quadratic_residue(N, p):
            factor_base.append(p)
    return factor_base

def b_smooth_factor(x: int, factor_base: list) -> dict | None:
    """Trial division to check if x is B-Smooth and if so to find factorization

        :param x: The number to factor
        :param factor_base: The factor_base
        :returns powers: A dictionary of the form {p: exponent} if x is B-smooth, None otherwise
    """
    powers = dict()
    if x < 0:
        powers[-1] = 1
        x = -x
    for p in factor_base[1:]:
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


def process_one_interval(N: int, factor_base: list[int], dict_factor_base: dict[int, int], log_factor_base: list[float], iteration: int, sieve_interval_size: int):
    """The exact same functionality as the better_find_b_smooth_squares function, but parallelized. 
       Check the docstring for that function for more information"""

    sieve_interval_center = math.ceil(math.sqrt(N))
    small_start = sieve_interval_center - (iteration+1) * sieve_interval_size
    large_start = sieve_interval_center + iteration * sieve_interval_size
    smaller_registers = np.zeros(sieve_interval_size)
    larger_registers = np.zeros(sieve_interval_size)

    for p in factor_base[1+PRIMES_TO_SKIP:]:
        sols = sqrt_mod(N, p, all_roots=True)
        for sol in sols:
            for i in range((sol-large_start) % p, sieve_interval_size, p):
                larger_registers[i] += log_factor_base[dict_factor_base[p]]
            for i in range((sol-small_start) % p, sieve_interval_size, p):
                smaller_registers[i] += log_factor_base[dict_factor_base[p]]

    results = []
    for i in range(0, sieve_interval_size):
        for start, regs in [(small_start, smaller_registers), (large_start, larger_registers)]:
            x = i + start
            fx = x**2 - N
            if regs[i] >= math.log(abs(fx)) - SIEVE_CUSHION:
                factors = b_smooth_factor(fx, factor_base)
                if factors is not None:
                    results.append((x, factors))
    return results
   
def parallel_find_b_smooth_squares(N: int, factor_base: list[int], dict_factor_base: dict[int, int], verbose: bool) -> list:
    """Just a parallel version of the better_find_b_smooth_squares function which 
       processes each interval in parallel. Functionality is the same as better_find_b_smooth_squares.
       Check the docstring for that function for more information"""
    log_factor_base  = [math.log(p) if p != -1 else 0 for p in factor_base]
    nums = []
    factorizations = []
    sieve_interval_size = min(SIEVE_INTERVAL_SIZE, int(math.sqrt(N)) - 2)

    max_iters = (int(math.sqrt(N)) // sieve_interval_size) - 1
    vectors_needed = len(factor_base) + 1 + EXTRA_VECTORS

    start_number = os.cpu_count()   # let's just start with the number of iterations our computer can do in parallel

    with ProcessPoolExecutor() as executor:
        futures = []
        # submit the first start_number iterations
        for i in range(0, start_number):
            futures.append(executor.submit(process_one_interval, N, factor_base, dict_factor_base, log_factor_base, i, sieve_interval_size))
        iteration = start_number


        while futures and len(nums) < vectors_needed:
            for future in as_completed(futures):    # as we complete iterations
                result = future.result()
                # add the results from the interval to our list
                for x, factors in result:
                    nums.append(x)
                    factorizations.append(factors)
                    if verbose and len(nums) % UPDATE_INTERVAL == 0:
                            print(f"Found {len(nums)} of {len(factor_base) + 1 + EXTRA_VECTORS} needed B-smooth numbers so far")

                futures.remove(future)      # remove the iteration we just finished

                # if we have more work to do, submit the next iteration
                if iteration < max_iters and len(nums) < vectors_needed:
                    futures.append(executor.submit(process_one_interval, N, factor_base, dict_factor_base, log_factor_base, iteration, sieve_interval_size))
                    iteration += 1

                # if we have enough numbers, break
                if len(nums) >= vectors_needed:
                    break
                
    return nums, factorizations


def better_find_b_smooth_squares(N: int, factor_base: list[int], dict_factor_base: dict[int, int], verbose: bool) -> list:
    """Finds B-smooth squares mod N using the quadratic sieve method

        :param N: The number to factor
        :param B: The bound for the factor base
        :param factor_base: The factor base
        :param dict_factor_base: A dictionary of the form {p: index} for the factor base
        :returns nums: A list of numbers that are B-smooth squares mod N
        :returns factorizations: A list of dictionaries of the form {p: exponent} for each number in nums
    """
    
    log_factor_base  = [math.log(p) if p != -1 else 0 for p in factor_base] # take the log of each of the primes except -1
    sieve_interval_size = min(SIEVE_INTERVAL_SIZE, int(math.sqrt(N)) - 2)   # don't want to try and sieve any negative numbers
    sieve_interval_center = math.ceil(math.sqrt(N))
    iteration = 0
    polynomial_solutions = [sqrt_mod(N, p, all_roots=True) for p in factor_base] # the solutions to the polynomial mod p

    nums = []               # the numbers we will return (note we return the numbers themselves, not their squares mod N)
    factorizations = []     # the factorizations of the numbers squared mod N

    # we want enough numbers to guarantee a linear dependence
    vectors_needed = len(factor_base) + 1 + EXTRA_VECTORS
    while len(nums) < vectors_needed:
        # the true x value of the first x we will check in both the small and large intervals
        small_start = sieve_interval_center - (iteration+1) * sieve_interval_size
        large_start = sieve_interval_center + iteration * sieve_interval_size

        smaller_registers = np.zeros(sieve_interval_size)     # for numbers below sqrt(N)
        larger_registers = np.zeros(sieve_interval_size)      # for numbers above sqrt(N)
        
        # for each prime in the factor base (we can skip the first few since their log is small), we will find the two solutions to x^2 = N mod p
        for i, p in enumerate(factor_base):
            if i < PRIMES_TO_SKIP + 1:   # skip the first few primes
                continue
            sols = polynomial_solutions[dict_factor_base[p]]   # the solutions to the polynomial mod p
            for sol in sols:
                for i in range((sol-large_start) % p, sieve_interval_size, p):   # all numbers congruent to sol mod p are also solutions
                    larger_registers[i] += log_factor_base[dict_factor_base[p]]    # so we add the log of the prime to the register

                for i in range((sol-small_start) % p, sieve_interval_size, p):   # same as above for the smaller registers
                    smaller_registers[i] += log_factor_base[dict_factor_base[p]]    
        
        # for numbers x whose sum of logs of prime divisors is close to the number, we check if they are B-smooth
        for i in range(0, sieve_interval_size):
            for start, regs in [(small_start, smaller_registers), (large_start, larger_registers)]:     # check both the smaller and larger registers
                x = i + start
                fx = x**2 - N   # the polynomial value at x
                if regs[i] >= math.log(abs(fx)) - SIEVE_CUSHION:       # this formula is what we define as "close enough" in the textbook
                    factors = b_smooth_factor(fx, factor_base)      # try to factor by trial division
                    if factors is not None:
                        # if the number is B smooth, we will add it to our list
                        nums.append(x)
                        factorizations.append(factors)
                        if verbose and len(nums) % UPDATE_INTERVAL == 0:
                            print(f"Found {len(nums)} of {vectors_needed} needed B-smooth numbers so far")

                        if len(nums) >= vectors_needed:   # we have found enough factors
                            break

        # on the next iteration, we will search further away from sqrt(N)
        iteration += 1
        if (iteration+1) * sieve_interval_size > math.sqrt(N):  # we've exhausted [0, 2sqrt(N)]
            break
    return nums, factorizations     # if we didn't find enough numbers, we will just return what we have and maybe get lucky


def vectorize_factorizations(factorizations: dict[int, int], dict_factor_base: dict[int, int]) -> np.array:
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

def quadratic_sieve(N: int, verbose: bool=False):
    """Quadratic Sieve implementation to factor N

        :param N: The number to factor
        :param verbose: If True, print debug information
        :returns factors: A tuple of the two factors if found, None if it can't find any"""
    found_factors = False
    B = choose_factor_base_bound(N)
    while not found_factors and B < math.sqrt(N):
    
        factor_base = list(build_smart_factor_base(N, B))
        if verbose:
            print(f"=== Chose factor base of {B} consisting of {len(factor_base)} primes===")
        dict_factor_base = {p: i for i, p in enumerate(factor_base)}

        if verbose:
            print("=== Searching for B-smooth squares ===")
        
        if PARALLEL:
            nums, factorizations = parallel_find_b_smooth_squares(N, factor_base, dict_factor_base, verbose)
        else:
            nums, factorizations = better_find_b_smooth_squares(N, factor_base, dict_factor_base, verbose)

        if nums == []:
            B *= 2
            continue
        
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
        
        # if we didn't find a solution, let's double B and try again
        if verbose:
            print("=== No factors found, trying again with larger B ===")
        B *=2

    return None

if __name__ == '__main__':
    # Number to factor
    N = int(sys.argv[1])
    time_start = time.time()
    quadratic_sieve(N, verbose=True)
    time_end = time.time()
    print("Time elapsed: ", round(time_end - time_start, 2), "seconds")
