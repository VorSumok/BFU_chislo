import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, simplify, diff, init_printing

x = symbols('x')
init_printing(use_unicode=True)

np.set_printoptions(precision=7, suppress=True)


class Function:
    def __init__(self, start, end, function, eps=0.0000001):
        self.a = start
        self.b = end
        self.show_fun = simplify(function)
        self.fun = lambdify(x, simplify(function))
        self.df = lambdify(x, diff(simplify(function), x))
        self.ddf = lambdify(x, diff(simplify(function), x, 2))
        self.eps = eps

    def show_f_df_ddf(self):
        f = simplify(self.show_fun)
        df = simplify(diff(self.show_fun, x))
        ddf = simplify(diff(self.show_fun, x, 2))
        return f, df, ddf

    def newton_method(self):
        x0 = self.a if self.fun(self.a) * self.ddf(self.a) > 0 else self.b
        result = [(x0, 0)]
        n = 1
        while True:
            x1 = x0 - self.fun(x0) / self.df(x0)
            if abs(x1 - x0) < self.eps:
                break
            result.append((x1, n))
            x0 = x1
            n += 1
        return result

    def chord_method(self):
        x0, ab = (self.a, self.b) if self.fun(self.b) * self.ddf(self.b) > 0 else (self.b, self.a)
        result = [(x0, 0)]
        n = 1
        while True:
            x1 = x0 - self.fun(x0) * (ab - x0) / (self.fun(ab) - self.fun(x0))
            if abs(x1 - x0) < self.eps:
                break
            result.append((x1, n))
            x0 = x1
            n += 1
        return result

    def secant_method(self):
        x0 = self.a
        x1 = self.b
        result = [(x0, 0), (x1, 0)]
        n = 1
        while True:
            x2 = x1 - (self.fun(x1) * (x1 - x0) / (self.fun(x1) - self.fun(x0)))
            if abs(x2 - x1) < self.eps:
                break
            result.append((x2, n))
            x0 = x1
            x1 = x2
            n += 1
        return result

    def finite_difference_newton_method(self, step=1000):
        x0 = self.a
        result = [(x0, 0)]
        h = (self.b - self.a) / step
        n = 1
        while True:
            f1 = self.fun(x0) * h
            f2 = (self.fun(x0 + h) - self.fun(x0))
            x1 = x0 - f1 / f2
            if abs(x1 - x0) < self.eps:
                break
            result.append((x1, n))
            x0 = x1
            n += 1
        return result

    def steffensen_method(self):
        x0 = self.a
        result = [(x0, 0)]
        n = 1
        while True:
            x1 = x0 - self.fun(x0) ** 2 / (self.fun(x0 + self.fun(x0)) - self.fun(x0))
            if abs(x1 - x0) < self.eps:
                break
            result.append((x1, n))
            x0 = x1
            n += 1
        return result

    def simple_iteration_method(self):
        x0 = self.a
        result = [(x0, 0)]
        n = 1
        min_df = self.find_min_fun(1000, self.df)[0]
        if min_df > 0:
            T = 1 / min_df
            while True:
                x1 = x0 - T * self.fun(x0)
                if abs(x1 - x0) < self.eps:
                    break
                result.append((x1, n))
                x0 = x1
                n += 1
        return result

    def find_min_fun(self, n, fun):
        point = self.a
        min_fx = fun(point)
        for i in np.linspace(self.a, self.b, n):
            if min_fx > fun(i):
                min_fx, point = fun(i), i
        return [min_fx, point]

class GUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Калькулятор численного решения уравнений")

        self.function_label = tk.Label(self.window, text="Функция")
        self.function_label.grid(row=0, column=0, pady=10, padx=5)

        self.function_entry = tk.Entry(self.window, width=30)
        self.function_entry.grid(row=0, column=1, pady=10)
        self.function_entry.insert(tk.END, "exp(x) - 2*(x-2)**2")

        self.start_label = tk.Label(self.window, text="Начало диапазона")
        self.start_label.grid(row=1, column=0, pady=10, padx=5)

        self.start_entry = tk.Entry(self.window, width=10)
        self.start_entry.grid(row=1, column=1, pady=10)
        self.start_entry.insert(tk.END, "0")

        self.end_label = tk.Label(self.window, text="Конец диапазона")
        self.end_label.grid(row=1, column=2, pady=10, padx=5)

        self.end_entry = tk.Entry(self.window, width=10)
        self.end_entry.grid(row=1, column=3, pady=10)
        self.end_entry.insert(tk.END, "1")

        self.calculate_button = tk.Button(self.window, text="Рассчитать", command=self.calculate)
        self.calculate_button.grid(row=1, column=4, padx=10)

        self.derivatives_label = tk.Label(self.window, text="Производные")
        self.derivatives_label.grid(row=2, column=0, pady=10, padx=5)

        self.derivatives_output = tk.Text(self.window, width=30, height=2)
        self.derivatives_output.grid(row=2, column=1, columnspan=4, pady=10)

        self.methods_label = tk.Label(self.window, text="Методы")
        self.methods_label.grid(row=3, column=0, pady=10, padx=5)

        self.methods_output = tk.Text(self.window, width=80, height=20)
        self.methods_output.grid(row=3, column=1, columnspan=4, padx=10)

        self.graph_button = tk.Button(self.window, text="Построить график", command=self.plot_graph)
        self.graph_button.grid(row=4, column=1, columnspan=4, pady=10)

        self.window.mainloop()

    def calculate(self):
        # Получение данных из полей ввода
        function = self.function_entry.get()
        start = float(self.start_entry.get())
        end = float(self.end_entry.get())

        # Создание объекта класса Function
        fnc = Function(start, end, function)

        # Вывод производных
        f, df, ddf = fnc.show_f_df_ddf()
        derivatives = f"F'(x) = {df}\nF''(x) = {ddf}"
        self.derivatives_output.delete('1.0', tk.END)
        self.derivatives_output.insert(tk.END, derivatives)

        # Вычисление и вывод методов
        result_text = ""
        result_text += "\n--Ньютон--\n"
        result_text += self.format_results(fnc.newton_method())
        result_text += "\n--Погреность--\n"
        result_text += f"{(abs(fnc.fun(fnc.newton_method()[-1][0]) / fnc.find_min_fun(1000, fnc.df)[0]))}\n"

        result_text += "\n--Метод хорд--\n"
        result_text += self.format_results(fnc.chord_method())
        result_text += "\n--Погреность--\n"
        result_text += f"{(abs(fnc.fun(fnc.chord_method()[-1][0]) / fnc.find_min_fun(1000, fnc.df)[0]))} \n"

        result_text += "\n--Метод секущих--\n"
        result_text += self.format_results(fnc.secant_method())
        result_text += "\n--Погреность--\n"
        result_text += f"{(abs(fnc.fun(fnc.secant_method()[-1][0]) / fnc.find_min_fun(1000, fnc.df)[0]))}\n"

        result_text += "\n--Конечноразностный метод Ньютона--\n"
        result_text += self.format_results(fnc.finite_difference_newton_method())
        result_text += "\n--Погреность--\n"
        result_text += f"{(abs(fnc.fun(fnc.finite_difference_newton_method()[-1][0]) / fnc.find_min_fun(1000, fnc.df)[0]))}\n"

        result_text += "\n--Метод Стеффенсена--\n"
        result_text += self.format_results(fnc.steffensen_method())
        result_text += "\n--Погреность--\n"
        result_text += f"{(abs(fnc.fun(fnc.steffensen_method()[-1][0]) / fnc.find_min_fun(1000, fnc.df)[0]))}\n"

        result_text += "\n--Метод простых итераций--\n"
        result_text += self.format_results(fnc.simple_iteration_method())
        result_text += "\n--Погреность--\n"
        result_text += f"{(abs(fnc.fun(fnc.simple_iteration_method()[-1][0]) / fnc.find_min_fun(1000, fnc.df)[0]))}\n"

        self.methods_output.delete('1.0', tk.END)
        self.methods_output.insert(tk.END, result_text)

    def format_results(self, results):
        result_text = "Итерация  Значение\n"
        for iteration, value in results:
            result_text += f"{value:.0f} {iteration:.7f}\n"
        return result_text

    def plot_graph(self):
        # Получение данных из полей ввода
        function = self.function_entry.get()
        start = float(self.start_entry.get())
        end = float(self.end_entry.get())

        # Создание объекта класса Function
        fnc = Function(start, end, function)

        # Построение графика
        x_values = np.linspace(start, end, 100)
        f_values = fnc.fun(x_values)

        plt.figure()
        plt.plot(x_values, f_values, label='f(x) = ' + function)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'График функции на промежутке [{start}; {end}]')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    gui = GUI()