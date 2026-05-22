import numpy as np
import matplotlib.pyplot as plt
import sqlite3
conn = sqlite3.connect("housing.db")
cursor = conn.cursor()

cursor.execute("""
               CREATE TABLE IF NOT EXISTS housing (
                    size REAL,
                    price REAL
    )
""")
    
cursor.execute("DELETE FROM housing")
cursor.executemany("INSERT INTO housing VALUES (?, ?)", [(1.0, 300.0), (2.0, 500.0)])
conn.commit()

cursor.execute("SELECT size, price FROM housing")
data = cursor.fetchall()
x_train = np.array([row[0] for row in data])
y_train = np.array([row[1] for row in data])

def compute_cost(x, y, w, b):
    m = len(x)
    cost = 0
    for i in range (m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    return cost / (2 * m)

def compute_gradient(x, y,w, b):
    m = len(x)
    dw = 0
    db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dw += (f_wb - y[i]) * x[i]
        db += (f_wb - y[i])
    return dw / m, db / m

def gradient_descent(x, y, w, b, alpha, iterations):
    cost_history = []
    for i in range(iterations):
        dw, db = compute_gradient(x, y, w, b)
        w = w - alpha *dw
        b =b - alpha * db
        cost_history.append(compute_cost(x, y, w, b))
        if  i % 100 == 0:
            print(f"Iteration {i}: Cost {cost_history[-1]:.2f}. w = {w:.2f}, b = {b:.2f}")
    return w, b, cost_history 

w = 0
b = 0
alpha = 0.01
iterations = 1000

w_final, b_final, cost_history = gradient_descent(x_train, y_train, w, b, alpha, iterations)
print(f"\nFinal w = {w_final:.2f}, Final b = {b_final:.2f}")
    
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Gradient Descnet - Cost Going Down")
plt.show()
