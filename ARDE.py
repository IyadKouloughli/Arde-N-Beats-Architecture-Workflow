import random
import numpy as np

def initialization(D, bL, bU):
    """
    Initialize a solution vector within bounds.
    Args:
        D: Number of dimensions (e.g., number of variables to optimize).
        bL: Lower bounds for each dimension (e.g., [-5, -5, ...]).
        bU: Upper bounds for each dimension (e.g., [5, 5, ...]).
    Returns:
        z: Initialized vector [x1, x2, ..., xD].
    """
    z = []
    for i in range(D):
        z.append(bL[i] + random.uniform(0, 1) * (bU[i] - bL[i]))
    return z

def fitness(z):
    """
    Fitness function: Sum of variables (to be minimized).
    Args:
        z: Solution vector [x1, x2, ..., xD].
    Returns:
        float: Fitness value (sum of variables, lower is better).
    """
    return sum(z)  # Simple sum: f(x) = x1 + x2 + ... + xD

def reinitialization(z_candidates, P1, fitness_threshold):
    """
    Reinitialize population by selecting candidates with fitness below threshold.
    Args:
        z_candidates: Initial population (list of vectors).
        P1: Desired population size after reinitialization.
        fitness_threshold: Fitness threshold for selection.
    Returns:
        List of P1 selected candidates.
    """
    selected_candidates = []
    for z in z_candidates:
        if fitness(z) < fitness_threshold:
            selected_candidates.append(z)
            if len(selected_candidates) >= P1:
                break
    while len(selected_candidates) < P1:
        remaining = [z for z in z_candidates if z not in selected_candidates]
        if remaining:
            selected_candidates.append(random.choice(remaining))
        else:
            selected_candidates.append(initialization(D, bL, bU))
    return selected_candidates

def update(G, FL, FU):
    """
    Update mutation bounds FL and FU adaptively.
    Args:
        G: Current generation.
        FL: Current lower mutation factor.
        FU: Current upper mutation factor.
    Returns:
        Tuple (FL, FU): Updated mutation bounds.
    """
    if G > 0:
        FU_new = random.uniform(max(0.1, FL * 0.5), min(0.9, FL * 1.5))
        FL_new = random.uniform(0.1, FU_new)
        FL = FL_new
        FU = FU_new
    else:
        FL = 0.8
        FU = 1
    print(f"Generation {G}, FL: {FL:.4f}, FU: {FU:.4f}")
    return FL, FU

def mutation(z_new_candidates, FL, FU, bL, bU):
    """
    Perform differential mutation on candidates.
    Args:
        z_new_candidates: Current population.
        FL, FU: Mutation factor bounds.
        bL, bU: Parameter bounds.
    Returns:
        List of mutated vectors.
    """
    FG = random.uniform(FL, FU)
    mutated_candidates = []
    for z in z_new_candidates:
        selected_indices = random.sample(range(len(z_new_candidates)), 3)
        z_k, z_l, z_m = [z_new_candidates[i] for i in selected_indices]
        vG = []
        for i in range(len(z)):
            val = z_k[i] + FG * (z_l[i] - z_m[i])
            val = max(bL[i], min(bU[i], val))  # Ensure within bounds
            vG.append(val)
        mutated_candidates.append(vG)
    return mutated_candidates

def crossover(z_mutation, Cr, z_new_candidates):
    """
    Perform binomial crossover.
    Args:
        z_mutation: Mutated vectors.
        Cr: Crossover rate (0 to 1).
        z_new_candidates: Current population.
    Returns:
        List of crossover vectors.
    """
    crossover_candidates = []
    for z, vG in zip(z_new_candidates, z_mutation):
        uG = []
        for i in range(len(vG)):
            Ci = random.uniform(0, 1)
            uG.append(vG[i] if Ci >= Cr else z[i])
        crossover_candidates.append(uG)
    return crossover_candidates

def selection(z_new_candidates, u_crossover):
    """
    Perform greedy selection based on fitness.
    Args:
        z_new_candidates: Current population.
        u_crossover: Crossover vectors.
    Returns:
        List of selected vectors for next generation.
    """
    selection_candidates = []
    for z, uG in zip(z_new_candidates, u_crossover):
        selection_candidates.append(uG if fitness(uG) <= fitness(z) else z)
    return selection_candidates

def earlyStopping(best_fitness_history, k, tolerance=1e-5):
    """
    Check for early stopping if best fitness hasn't improved within tolerance over k generations.
    Args:
        best_fitness_history: List of best fitness values.
        k: Patience parameter.
        tolerance: Fitness change tolerance.
    Returns:
        Boolean: True if stopping condition met.
    """
    if len(best_fitness_history) < k:
        return False
    for i in range(1, k):
        if abs(best_fitness_history[-1] - best_fitness_history[-1-i]) > tolerance:
            return False
    return True

def ARDE(P0, P1, FL, FU, Cr, maxG, k, fitness, fitness_threshold, verbose=True):
    """
    Adaptive Reinitialized Differential Evolution (ARDE) for optimization.
    Args:
        P0: Initial population size.
        P1: Reinitialized population size (P1 <= P0).
        FL, FU: Mutation factor bounds.
        Cr: Crossover rate.
        maxG: Maximum generations.
        k: Early stopping patience.
        fitness: Fitness function to minimize.
        fitness_threshold: Threshold for reinitialization.
        verbose: If True, print progress.
    Returns:
        Tuple (best_solution, fitness_values_history).
    """
    if P1 > P0:
        raise ValueError("P1 must be <= P0")
    if any(bL[i] > bU[i] for i in range(D)):
        raise ValueError("Lower bounds must be <= upper bounds")
    
    best_fitness_history = []
    fitness_values_history = []
    for G in range(maxG):
        if G == 0:
            z = [initialization(D, bL, bU) for _ in range(P0)]
            z_new_candidates = reinitialization(z, P1, fitness_threshold)
            FL, FU = update(G, FL, FU)
            z_mutation = mutation(z_new_candidates, FL, FU, bL, bU)
            u_crossover = crossover(z_mutation, Cr, z_new_candidates)
            z_selection = selection(z_new_candidates, u_crossover)
        else:
            FL, FU = update(G, FL, FU)
            z_mutation = mutation(z_selection, FL, FU, bL, bU)
            u_crossover = crossover(z_mutation, Cr, z_selection)
            z_selection = selection(z_selection, u_crossover)
            fitness_values = [fitness(individual) for individual in z_selection]
            fitness_values_history.append(fitness_values)
            best_fitness = min(fitness_values)
            best_fitness_history.append(best_fitness)
            if verbose:
                print(f"Generation {G+1} - Best Fitness: {best_fitness:.4f}")
            if len(best_fitness_history) >= k and earlyStopping(best_fitness_history, k, tolerance=1e-5):
                if verbose:
                    print(f"Early stopping triggered at generation {G}")
                break
    best_solution = min(z_selection, key=lambda x: fitness(x))
    if verbose:
        print(f"Optimal fitness: {fitness(best_solution):.4f}")
    return best_solution, fitness_values_history

# Parameters for sum minimization
D = 3  # Number of dimensions (variables x1, x2, x3)
bL = [-5] * D  # Lower bounds for each variable
bU = [5] * D   # Upper bounds for each variable
P0 = 9         # Initial population size
P1 = 5         # Reinitialized population size
FL = 0.8       # Initial lower mutation factor
FU = 1.0       # Initial upper mutation factor
Cr = 0.5       # Crossover rate
maxG = 100     # Maximum generations
k = 5          # Early stopping patience
fitness_threshold = 10  # Threshold for reinitialization (sum < 10)

# Test run
z_candidates = [initialization(D, bL, bU) for _ in range(P0)]
print("Initial candidates:")
for z in z_candidates:
    print([round(x, 2) for x in z])

z_new_candidates = reinitialization(z_candidates, P1, fitness_threshold)
print("\nReinitialized candidates:")
for z in z_new_candidates:
    print([round(x, 2) for x in z])

z_mutation = mutation(z_new_candidates, FL, FU, bL, bU)
print("\nMutated candidates:")
for z in z_mutation:
    print([round(x, 2) for x in z])

u_crossover = crossover(z_mutation, Cr, z_new_candidates)
print("\nCrossover candidates:")
for z in u_crossover:
    print([round(x, 2) for x in z])

z_selection = selection(z_new_candidates, u_crossover)
print("\nSelected candidates:")
for z in z_selection:
    print([round(x, 2) for x in z])

optimal_solution, fitness_values_history = ARDE(P0, P1, FL, FU, Cr, maxG, k, fitness, fitness_threshold)
print("\nOptimal solution:", [round(x, 2) for x in optimal_solution])
print(f"Optimal fitness: {fitness(optimal_solution):.4f}")
for generation, fitness_values in enumerate(fitness_values_history):
    print(f"Generation {generation + 1} fitness values:", [round(f, 2) for f in fitness_values])
