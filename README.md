# 🧠 ARDE-N-BEATS: Adaptive Reinitialized Differential Evolution

Welcome to **ARDE-N-BEATS**, an implementation of a novel **Adaptive Reinitialized Differential Evolution (ARDE)** algorithm — a population-based metaheuristic optimization method. This project demonstrates how ARDE evolves a population of candidate solutions using dynamic mutation factors, reinitialization, and early stopping strategies.

> 🔬 This implementation uses a **sum minimization fitness function** for demonstration but can be extended to more complex optimization tasks.

---

## 🚀 Features

- ✅ **Adaptive Mutation Factor**: Adjusts mutation bounds dynamically each generation  
- 🔁 **Reinitialization Mechanism**: Focuses search by retaining high-quality solutions  
- 🧬 **Differential Mutation + Binomial Crossover**  
- ⏹️ **Greedy Selection** for elite survival  
- 🛑 **Early Stopping** when no improvement is detected  
- 📊 **Verbose Logging** to track progress and fitness history  

---

## 📂 Project Files

- `ARDE.py`: Core implementation of the ARDE-N-BEATS optimization algorithm.

---

## 🧠 How It Works

1. **Initialize Population** with `P0` candidates  
2. **Select Best `P1` Candidates** based on a fitness threshold  
3. **Mutate + Crossover** new solutions using adaptive control parameters  
4. **Select Elites** to survive to the next generation  
5. **Early Stop** if best fitness doesn't improve for `k` generations  

---

## ⚙️ Parameters

You can configure these directly inside `ARDE.py`:

| Parameter            | Description                                  | Example |
|----------------------|----------------------------------------------|---------|
| `D`                  | Problem dimensions                           | `3`     |
| `bL`, `bU`           | Lower and upper bounds                       | `[-5]`, `[5]` |
| `P0`, `P1`           | Initial and reinitialized population sizes   | `9`, `5` |
| `FL`, `FU`           | Mutation factor bounds                       | `0.8`, `1.0` |
| `Cr`                 | Crossover rate                               | `0.5`   |
| `maxG`               | Max number of generations                    | `100`   |
| `k`                  | Early stopping patience                      | `5`     |
| `fitness_threshold`  | Reinitialization threshold                   | `10`    |

---

## 📊 Optional: Plot Fitness Curve

To visualize how the best fitness improves over generations, add this snippet at the end of the script:

```python
import matplotlib.pyplot as plt
plt.plot([min(g) for g in fitness_values_history])
plt.title("Best Fitness Over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid(True)
plt.show()

---

## 🛠 Requirements

Python 3.x
NumPy
Matplotlib (optional, for plotting)

---

## 💬 Contact

Author: Iyad Kouloughli
📧 iyadkdb@gmail.com
