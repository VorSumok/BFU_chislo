import numpy as np
import tkinter as tk


def gaussian_elimination(matrix, vector):
    n = len(matrix)
    augmented_matrix = np.column_stack((matrix, vector))

    for i in range(n):
        pivot_row = max(range(i, n), key=lambda k: abs(augmented_matrix[k, i]))
        augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]

        for j in range(i + 1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]

    solution = np.zeros(n)
    for i in range(n - 1, -1, -1):
        solution[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:-1], solution[i + 1:])) / \
                      augmented_matrix[i, i]

    return solution


def seidel(matrix_a, vector_b, tol=10 ** (-15)):
    a = matrix_a.T.dot(matrix_a)
    b = matrix_a.T.dot(vector_b)
    x = np.zeros_like(b)
    while True:
        x_old = np.copy(x)
        for i in range(len(x)):
            x[i] = (b[i] - np.dot(a[i, :i], x[:i]) - np.dot(a[i, i + 1:], x_old[i + 1:])) / a[i, i]
        if np.linalg.norm(x - x_old) < tol:
            break
    return x


def on_enter(event):
    event.widget.tk_focusNext().focus()


class GUI:
    def __init__(self, root_name):
        self.root = root_name

        matrix_frame = tk.Frame(self.root)
        matrix_frame.pack(pady=10)

        matrix_label = tk.Label(matrix_frame, text="Матрица")
        matrix_label.grid(row=0, columnspan=4)

        default_matrix_values = [
            [0.74, -0.62, 2.11, 0.55],
            [0.50, 0.98, 1.79, 0.09],
            [-0.73, 0.25, 2.07, 1.00],
            [1.00, -0.85, 1.95, 0.15]
        ]

        self.matrix_entries = []
        for i in range(4):
            row = []
            for j in range(4):
                default_value = default_matrix_values[i][j]
                entry = tk.Entry(matrix_frame, width=8)
                entry.grid(row=i + 1, column=j)
                entry.insert(0, str(default_value))
                entry.bind("<Return>", on_enter)
                row.append(entry)
            self.matrix_entries.append(row)

        vector_frame = tk.Frame(root_name)
        vector_frame.pack()

        vector_label = tk.Label(vector_frame, text="Вектор")
        vector_label.pack()

        default_vector_b_values = [3.18, 0.56, -2.89, 5.20]

        self.vector_entries_b = []
        for i in range(4):
            default_value = default_vector_b_values[i]
            entry = tk.Entry(vector_frame, width=8)
            entry.pack(side=tk.LEFT)
            entry.insert(0, str(default_value))
            entry.bind("<Return>", on_enter)
            self.vector_entries_b.append(entry)

        exact_solution_frame = tk.Frame(root_name)
        exact_solution_frame.pack()

        exact_solution_label = tk.Label(exact_solution_frame, text="Точное решение")
        exact_solution_label.pack()

        default_vector_c_values = [2, -2, 1, -3]

        self.vector_entries_c = []
        for i in range(4):
            default_value = default_vector_c_values[i]
            entry = tk.Entry(exact_solution_frame, width=8)
            entry.pack(side=tk.LEFT)
            entry.insert(0, str(default_value))
            entry.bind("<Return>", on_enter)
            self.vector_entries_c.append(entry)

        buttons_frame = tk.Frame(root_name)
        buttons_frame.pack()

        solve_button = tk.Button(buttons_frame, text="Решить уравнения", command=self.solve_equations)
        solve_button.pack(side=tk.LEFT, padx=10)

        clear_button = tk.Button(buttons_frame, text="Очистить",
                                 command=lambda: [enter.delete(0, tk.END) for rowed in self.matrix_entries for enter in
                                                  rowed] +
                                                 [entry_b.delete(0, tk.END) for entry_b in self.vector_entries_b] +
                                                 [entry_c.delete(0, tk.END) for entry_c in self.vector_entries_c])
        clear_button.pack(side=tk.LEFT)

        self.result_label_gaussian = tk.Label(root_name, text="")
        self.result_label_gaussian.pack()
        self.error_label_gaussian = tk.Label(root_name, text="")
        self.error_label_gaussian.pack()

        self.result_label_seidel = tk.Label(root_name, text="")
        self.result_label_seidel.pack()
        self.error_label_seidel = tk.Label(root_name, text="")
        self.error_label_seidel.pack()

    def solve_equations(self):
        input_matrix = [[float(entry.get()) for entry in row] for row in self.matrix_entries]
        matrix_a = np.array(input_matrix)

        input_vector_b = [float(entry.get()) for entry in self.vector_entries_b]
        vector_b = np.array(input_vector_b)

        input_vector_c = [float(entry.get()) for entry in self.vector_entries_c]
        vector_c = np.array(input_vector_c)

        solution_gaussian = gaussian_elimination(matrix_a, vector_b)
        error_gaussian = np.linalg.norm(solution_gaussian - vector_c)

        solution_seidel = seidel(matrix_a, vector_b)
        error_seidel = np.linalg.norm(solution_seidel - vector_c)

        self.result_label_gaussian.config(
            text="Решение методом Гаусса: " + ", ".join("{:.8f}".format(x) for x in solution_gaussian))
        self.error_label_gaussian.config(text="Погрешность: " + str(error_gaussian))
        self.result_label_seidel.config(
            text="Решение методом Зейделя: " + ", ".join("{:.16f}".format(x) for x in solution_seidel))
        self.error_label_seidel.config(text="Погрешность: " + str(error_seidel))


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Решение линейных уравнений")
    root.geometry("650x320")

    gui = GUI(root)

    root.mainloop()
