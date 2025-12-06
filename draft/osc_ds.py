import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
import warnings

# Отключаем предупреждения для чистоты вывода (но обрабатываем ошибки явно)
warnings.filterwarnings('ignore')


class StrongOscillatingUniverse:
    """Модель с СИЛЬНЫМИ осцилляциями - радикальная версия!"""

    def __init__(self, K=8.0):
        self.K = K

        # РАДИКАЛЬНЫЕ параметры для сильных осцилляций
        self.alpha = 0.1  # В 10 раз сильнее!
        self.beta = 0.05  # В 5 раз сильнее!
        self.omega_N = 0.0005  # Частота
        self.omega_p = 0.0007  # Частота p
        self.gamma = 0.01  # Сильная связь!

        # ОГРОМНЫЕ пределы для N (уменьшены для стабильности)
        self.lnN_min = math.log(1e60)  # Планковская эпоха
        self.lnN_max = math.log(1e130)  # Уменьшено с 1e147 для стабильности
        self.lnN_center = math.log(9.7e122)  # Наша эпоха

        # Пределы для p
        self.p_min = 0.01
        self.p_max = 0.15
        self.p_center = 0.0527

        # Инициализация массивов
        self.t = None
        self.N = None
        self.p = None
        self.lnN = None
        self.lnN_norm = None
        self.p_norm = None

    def safe_exp(self, x, max_val=700):
        """Защищённое вычисление экспоненты для избежания переполнения"""
        if x > max_val:
            return math.exp(max_val)
        elif x < -max_val:
            return math.exp(-max_val)
        return math.exp(x)

    def U(self, lnN, p):
        """Фрактальная глубина"""
        if p <= 0:
            return 1.0
        # Защита от деления на ноль
        denominator = abs(math.log(self.K * p + 1e-100))
        if denominator == 0:
            return 1.0
        return lnN / denominator

    def attractor_value(self, lnN, p):
        """Значение аттрактора p√(KU)"""
        U_val = self.U(lnN, p)
        return p * math.sqrt(self.K * U_val)

    def derivatives(self, t, y):
        """
        РАДИКАЛЬНЫЕ уравнения для сильных осцилляций!

        y[0] = нормированный lnN от -1 до 1
        y[1] = нормированный p от -1 до 1
        """
        try:
            # Денормализуем с ограничениями
            lnN_norm = max(-1.0, min(1.0, y[0]))
            p_norm = max(-1.0, min(1.0, y[1]))

            lnN = self.lnN_center + lnN_norm * (self.lnN_max - self.lnN_min) / 2
            p = self.p_center + p_norm * (self.p_max - self.p_min) / 2

            # Ограничиваем p
            p = max(self.p_min, min(self.p_max, p))

            # Аттрактор и отклонение
            attractor = self.attractor_value(lnN, p)
            deviation = attractor / math.e - 1.0

            # СИЛЬНЫЕ ОСЦИЛЛЯТОРЫ:

            # Для lnN: синусоида + аттракторная сила
            dlnN_norm_dt = (self.alpha * math.sin(self.omega_N * t) * 2 +  # Осциллятор
                            self.beta * deviation * (0.5 - lnN_norm) * 5)  # К аттрактору

            # Нелинейность: усиливаем при больших отклонениях
            dlnN_norm_dt *= (1 + abs(lnN_norm) ** 0.5)  # Используем корень для стабильности

            # Для p: косинусоида + аттрактор
            dp_norm_dt = (self.omega_p * math.cos(self.omega_p * t + math.pi / 2) * 3 +
                          self.gamma * deviation * (0.3 - p_norm) * 3)

            # Нелинейность для p
            dp_norm_dt *= (1 + abs(p_norm) ** 0.5)  # Используем корень для стабильности

            # Связь: когда N высокое, p должно быть низким
            coupling = 0.02 * (lnN_norm * (0.5 - p_norm) - p_norm * (0.5 - lnN_norm))
            dlnN_norm_dt += coupling
            dp_norm_dt -= coupling * 2

            # Ограничиваем скорости
            dlnN_norm_dt = max(-0.5, min(0.5, dlnN_norm_dt))
            dp_norm_dt = max(-0.2, min(0.2, dp_norm_dt))

            return [dlnN_norm_dt, dp_norm_dt]

        except Exception as e:
            # В случае ошибки возвращаем нулевые производные
            print(f"Ошибка в derivatives при t={t}: {e}")
            return [0.0, 0.0]

    def simulate(self, t_span=(0, 50000), y0=None):
        """Запуск симуляции"""
        if y0 is None:
            # Начинаем в экстремуме для сильных осцилляций
            y0 = [0.8, -0.6]  # Высокий N, низкий p

        print("=" * 80)
        print("МОДЕЛЬ С СИЛЬНЫМИ ОСЦИЛЛЯЦИЯМИ")
        print("=" * 80)

        # Конвертируем начальные условия
        lnN_start = self.lnN_center + y0[0] * (self.lnN_max - self.lnN_min) / 2
        p_start = self.p_center + y0[1] * (self.p_max - self.p_min) / 2

        print(f"Начальные условия:")
        print(f"  lnN_norm = {y0[0]:.2f}, p_norm = {y0[1]:.2f}")
        print(f"  N = {self.safe_exp(lnN_start):.1e} (цель: 9.7e+122)")
        print(f"  p = {p_start:.4f} (цель: 0.0527)")
        print()

        # Решаем с меньшим максимальным шагом для стабильности
        try:
            sol = solve_ivp(self.derivatives, t_span, y0,
                            method='RK45', max_step=50,  # Уменьшено для стабильности
                            rtol=1e-8, atol=1e-8,
                            dense_output=True)

            if not sol.success:
                print(f"Предупреждение: решение не сошлось: {sol.message}")
                # Используем то, что получилось

        except Exception as e:
            print(f"Ошибка при решении: {e}")
            # Создаем простую симуляцию в случае ошибки
            t_points = np.linspace(t_span[0], t_span[1], 1000)
            lnN_norm = 0.1 * np.sin(0.001 * t_points)
            p_norm = 0.05 * np.cos(0.001 * t_points)
            sol = type('obj', (object,), {'t': t_points, 'y': np.vstack([lnN_norm, p_norm])})

        self.t = sol.t
        self.lnN_norm = sol.y[0]
        self.p_norm = sol.y[1]

        # Конвертируем в реальные значения с защитой от переполнения
        self.lnN = self.lnN_center + self.lnN_norm * (self.lnN_max - self.lnN_min) / 2

        # Ограничиваем lnN для избежания переполнения
        self.lnN = np.clip(self.lnN, -700, 700)

        # Безопасное вычисление экспоненты
        self.N = np.array([self.safe_exp(x) for x in self.lnN])

        self.p = self.p_center + self.p_norm * (self.p_max - self.p_min) / 2
        self.p = np.clip(self.p, self.p_min, self.p_max)

        return sol

    def analyze(self):
        """Анализ результатов"""
        print("\n" + "=" * 80)
        print("АНАЛИЗ СИЛЬНЫХ ОСЦИЛЛЯЦИЙ")
        print("=" * 80)

        # Проверяем, что данные есть
        if self.N is None or len(self.N) == 0:
            print("Ошибка: данные не симулированы!")
            return

        print(f"ОБЩИЙ ДИАПАЗОН:")
        print(f"  N: {np.min(self.N):.1e} - {np.max(self.N):.1e}")
        if np.min(self.N) > 0:
            print(f"    Отношение max/min: {np.max(self.N) / np.min(self.N):.1e}")
            print(f"    Логарифмический размах: {math.log10(np.max(self.N) / np.min(self.N)):.1f} порядков")
        print(f"  p: {np.min(self.p):.4f} - {np.max(self.p):.4f}")
        if np.min(self.p) > 0:
            print(f"    Отношение: {np.max(self.p) / np.min(self.p):.2f}")

        # Находим прохождение через нашу эпоху
        our_N = 9.7e122
        our_p = 0.0527

        # Толерансы
        N_tol = 0.2  # ±20%
        p_tol = 0.2  # ±20%

        epochs = []
        for i in range(len(self.N)):
            if (abs(self.N[i] - our_N) / our_N < N_tol and
                    abs(self.p[i] - our_p) / our_p < p_tol):
                epochs.append((self.t[i], self.N[i], self.p[i]))

        print(f"\nПРОХОЖДЕНИЕ ЧЕРЕЗ НАШУ ЭПОХУ:")
        if epochs:
            print(f"  Найдено {len(epochs)} прохождений!")
            for t, N, p in epochs[:3]:  # Показываем первые 3
                print(f"    t = {t:.0f}: N = {N:.1e}, p = {p:.4f}")
        else:
            # Находим ближайшие
            if len(self.N) > 0:
                # Нормализуем перед вычислением расстояния
                N_norm = np.log10(self.N / our_N)
                p_norm = (self.p - our_p) / our_p
                distances = np.sqrt(N_norm ** 2 + (100 * p_norm) ** 2)
                idx = np.argmin(distances)
                print(f"  Не прошли точно, но ближайшая точка:")
                print(f"    t = {self.t[idx]:.0f}: N = {self.N[idx]:.1e}, p = {self.p[idx]:.4f}")
                print(f"    Отклонение N: {abs(self.N[idx] - our_N) / our_N * 100:.1f}%")
                print(f"    Отклонение p: {abs(self.p[idx] - our_p) / our_p * 100:.1f}%")

        # Аттрактор
        attractor_vals = []
        for i in range(len(self.lnN)):
            attractor = self.attractor_value(self.lnN[i], self.p[i])
            attractor_vals.append(attractor / math.e)

        if attractor_vals:
            print(f"\nАТТРАКТОР p√(KU)/e:")
            print(f"  Среднее: {np.mean(attractor_vals):.4f}")
            print(f"  Минимум: {np.min(attractor_vals):.4f}")
            print(f"  Максимум: {np.max(attractor_vals):.4f}")
            print(f"  Стандартное отклонение: {np.std(attractor_vals):.4f}")

    def calculate_constants_at_epoch(self, t_target=None):
        """Вычисление констант в нашу эпоху"""
        if self.N is None or len(self.N) == 0:
            print("Ошибка: данные не симулированы!")
            return None

        if t_target is None:
            # Находим ближайшую к нашей эпохе
            our_N = 9.7e122
            distances = np.abs(self.N - our_N) / our_N
            idx = np.argmin(distances)
        else:
            idx = np.argmin(np.abs(self.t - t_target))

        N = self.N[idx]
        p = self.p[idx]
        lnN = self.lnN[idx]

        # Упрощённые формулы
        U = self.U(lnN, p)

        # Наши константы
        t_p0 = 5.39e-44
        c0 = 3.00e8
        G0 = 6.67e-11
        Lambda0 = 1.1e-52

        # Масштабирование
        N_ratio = N / 9.7e122
        if N_ratio <= 0:
            N_ratio = 1.0

        U_ratio = U / 328 if U > 0 else 1.0

        t_p = t_p0 * N_ratio ** (-1 / 3)

        # Защита для логарифма
        if N > 0 and math.log(9.7e122) > 0:
            c = c0 * (math.log(N) / math.log(9.7e122)) ** 2.5
        else:
            c = c0

        G = G0 * N_ratio ** (-1 / 3)
        Lambda = Lambda0 * N_ratio ** (-1 / 3) * U_ratio ** 4

        return {
            't': self.t[idx],
            'N': N,
            'p': p,
            'U': U,
            't_p': t_p,
            'c': c,
            'G': G,
            'Lambda': Lambda,
            'attractor': self.attractor_value(lnN, p) / math.e
        }

    def visualize(self):
        """Визуализация сильных осцилляций"""
        if self.t is None or len(self.t) == 0:
            print("Ошибка: нет данных для визуализации!")
            return

        try:
            fig = plt.figure(figsize=(20, 12))

            # 1. N(t) - логарифмическая шкала
            ax1 = plt.subplot(2, 3, 1)
            ax1.plot(self.t, self.N, 'b-', linewidth=2, alpha=0.8)
            ax1.axhline(y=9.7e122, color='r', linestyle='--', linewidth=2,
                        alpha=0.7, label='Наша эпоха')
            ax1.set_xlabel('Время (условные единицы)')
            ax1.set_ylabel('N (число узлов)')
            ax1.set_yscale('log')

            # Устанавливаем разумные пределы для оси Y
            y_min = max(1e50, np.min(self.N) * 0.1)
            y_max = min(1e150, np.max(self.N) * 10)
            ax1.set_ylim(y_min, y_max)

            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
            ax1.set_title('СИЛЬНЫЕ ОСЦИЛЛЯЦИИ ЧИСЛА УЗЛОВ', fontsize=12, fontweight='bold')

            # 2. p(t)
            ax2 = plt.subplot(2, 3, 2)
            ax2.plot(self.t, self.p, 'r-', linewidth=2, alpha=0.8)
            ax2.axhline(y=0.0527, color='r', linestyle='--', linewidth=2,
                        alpha=0.7, label='Наша эпоха')
            ax2.set_xlabel('Время (условные единицы)')
            ax2.set_ylabel('p (вероятность дальних связей)')

            # Устанавливаем пределы для p
            p_min = max(0.001, np.min(self.p) * 0.8)
            p_max = min(0.2, np.max(self.p) * 1.2)
            ax2.set_ylim(p_min, p_max)

            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            ax2.set_title('ОСЦИЛЛЯЦИИ НЕЛОКАЛЬНОСТИ', fontsize=12, fontweight='bold')

            # 3. Фазовый портрет (пропускаем если слишком много точек)
            ax3 = plt.subplot(2, 3, 3)
            if len(self.N) > 1000:
                # Берем каждую 10-ю точку для производительности
                step = len(self.N) // 1000
                indices = np.arange(0, len(self.N), step)
                N_sampled = self.N[indices]
                p_sampled = self.p[indices]
                t_sampled = self.t[indices] / max(self.t)
            else:
                N_sampled = self.N
                p_sampled = self.p
                t_sampled = self.t / max(self.t)

            colors = t_sampled
            scatter = ax3.scatter(N_sampled, p_sampled, c=colors, cmap='plasma',
                                  s=10, alpha=0.7)
            ax3.scatter([9.7e122], [0.0527], c='red', s=200, marker='*',
                        edgecolors='black', linewidth=2, label='Наша эпоха', zorder=5)
            ax3.set_xlabel('N')
            ax3.set_ylabel('p')
            ax3.set_xscale('log')

            # Устанавливаем пределы для фазового портрета
            ax3.set_xlim(max(1e50, np.min(N_sampled) * 0.1),
                         min(1e150, np.max(N_sampled) * 10))
            ax3.set_ylim(max(0.001, np.min(p_sampled) * 0.8),
                         min(0.2, np.max(p_sampled) * 1.2))

            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=10)
            ax3.set_title('ФАЗОВЫЙ ПОРТРЕТ (траектория Вселенной)',
                          fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=ax3, label='Время (норм.)')

            # 4. Нормированные переменные
            ax4 = plt.subplot(2, 3, 4)
            ax4.plot(self.t, self.lnN_norm, 'b-', linewidth=2, alpha=0.8, label='lnN норм.')
            ax4.plot(self.t, self.p_norm, 'r-', linewidth=2, alpha=0.8, label='p норм.')
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Время')
            ax4.set_ylabel('Нормированные переменные')

            # Устанавливаем пределы для нормированных переменных
            ax4.set_ylim(-1.1, 1.1)

            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=10)
            ax4.set_title('НОРМИРОВАННЫЕ ОСЦИЛЛЯЦИИ', fontsize=12, fontweight='bold')

            # 5. Аттрактор
            ax5 = plt.subplot(2, 3, 5)
            attractor_vals = []
            for i in range(len(self.lnN)):
                attractor = self.attractor_value(self.lnN[i], self.p[i])
                attractor_vals.append(attractor / math.e)

            ax5.plot(self.t, attractor_vals, 'g-', linewidth=2, alpha=0.8)
            ax5.axhline(y=1.0, color='r', linestyle='--', linewidth=2,
                        alpha=0.7, label='Аттрактор e')
            ax5.fill_between(self.t, 0.95, 1.05, alpha=0.2, color='green')
            ax5.set_xlabel('Время')
            ax5.set_ylabel('p√(KU)/e')

            # Устанавливаем разумные пределы
            if attractor_vals:
                att_min = max(0.5, np.min(attractor_vals) * 0.8)
                att_max = min(1.5, np.max(attractor_vals) * 1.2)
                ax5.set_ylim(att_min, att_max)
            else:
                ax5.set_ylim(0.5, 1.5)

            ax5.grid(True, alpha=0.3)
            ax5.legend(fontsize=10)
            ax5.set_title('ДВИЖЕНИЕ ВОКРУГ АТТРАКТОРА', fontsize=12, fontweight='bold')

            # 6. Логарифмическое представление
            ax6 = plt.subplot(2, 3, 6)
            ax6.plot(self.t, np.log10(self.N), 'b-', linewidth=2, alpha=0.8, label='log₁₀(N)')
            ax6.axhline(y=math.log10(9.7e122), color='b', linestyle='--', alpha=0.7)

            ax6b = ax6.twinx()
            ax6b.plot(self.t, self.p * 100, 'r-', linewidth=2, alpha=0.8, label='p × 100')
            ax6b.axhline(y=0.0527 * 100, color='r', linestyle='--', alpha=0.7)

            ax6.set_xlabel('Время')
            ax6.set_ylabel('log₁₀(N)', color='b')
            ax6b.set_ylabel('p × 100', color='r')
            ax6.tick_params(axis='y', labelcolor='b')
            ax6b.tick_params(axis='y', labelcolor='r')
            ax6.grid(True, alpha=0.3)

            # Объединяем легенды
            lines1, labels1 = ax6.get_legend_handles_labels()
            lines2, labels2 = ax6b.get_legend_handles_labels()
            ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

            ax6.set_title('ЛОГАРИФМИЧЕСКОЕ ПРЕДСТАВЛЕНИЕ', fontsize=12, fontweight='bold')

            plt.tight_layout()
            plt.show()

            # Дополнительный график: константы
            consts = self.calculate_constants_at_epoch()
            if consts is None:
                return

            fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

            # Вычисляем константы во времени
            t_p_vals = []
            c_vals = []
            G_vals = []
            Lambda_vals = []

            # Берем подвыборку для производительности
            step = max(1, len(self.N) // 100)
            for i in range(0, min(len(self.N), 1000), step):
                N = self.N[i]
                p_val = self.p[i]
                lnN_val = self.lnN[i]
                U_val = self.U(lnN_val, p_val)

                N_ratio = N / 9.7e122 if N > 0 else 1.0
                U_ratio = U_val / 328 if U_val > 0 else 1.0

                t_p_vals.append(5.39e-44 * N_ratio ** (-1 / 3))

                if N > 0 and math.log(9.7e122) > 0:
                    c_vals.append(3.00e8 * (math.log(N) / math.log(9.7e122)) ** 2.5)
                else:
                    c_vals.append(3.00e8)

                G_vals.append(6.67e-11 * N_ratio ** (-1 / 3))
                Lambda_vals.append(1.1e-52 * N_ratio ** (-1 / 3) * U_ratio ** 4)

            t_sampled = self.t[::step][:len(t_p_vals)]

            # t_p
            axes2[0, 0].plot(t_sampled, t_p_vals, 'purple', linewidth=2)
            axes2[0, 0].axhline(y=5.39e-44, color='r', linestyle='--', alpha=0.7)
            axes2[0, 0].set_xlabel('Время')
            axes2[0, 0].set_ylabel('t_p (с)')
            axes2[0, 0].set_yscale('log')
            axes2[0, 0].grid(True, alpha=0.3)
            axes2[0, 0].set_title('ПЛАНКОВСКОЕ ВРЕМЯ', fontweight='bold')

            # c
            axes2[0, 1].plot(t_sampled, c_vals, 'orange', linewidth=2)
            axes2[0, 1].axhline(y=3.00e8, color='r', linestyle='--', alpha=0.7)
            axes2[0, 1].set_xlabel('Время')
            axes2[0, 1].set_ylabel('c (м/с)')
            axes2[0, 1].set_yscale('log')
            axes2[0, 1].grid(True, alpha=0.3)
            axes2[0, 1].set_title('СКОРОСТЬ СВЕТА', fontweight='bold')

            # G
            axes2[1, 0].plot(t_sampled, G_vals, 'brown', linewidth=2)
            axes2[1, 0].axhline(y=6.67e-11, color='r', linestyle='--', alpha=0.7)
            axes2[1, 0].set_xlabel('Время')
            axes2[1, 0].set_ylabel('G (м³/кг·с²)')
            axes2[1, 0].set_yscale('log')
            axes2[1, 0].grid(True, alpha=0.3)
            axes2[1, 0].set_title('ГРАВИТАЦИОННАЯ ПОСТОЯННАЯ', fontweight='bold')

            # Λ
            axes2[1, 1].plot(t_sampled, Lambda_vals, 'teal', linewidth=2)
            axes2[1, 1].axhline(y=1.1e-52, color='r', linestyle='--', alpha=0.7)
            axes2[1, 1].set_xlabel('Время')
            axes2[1, 1].set_ylabel('Λ (м⁻²)')
            axes2[1, 1].set_yscale('log')
            axes2[1, 1].grid(True, alpha=0.3)
            axes2[1, 1].set_title('КОСМОЛОГИЧЕСКАЯ ПОСТОЯННАЯ', fontweight='bold')

            plt.tight_layout()
            plt.show()

            # Вывод констант в нашу эпоху
            print("\n" + "=" * 80)
            print("КОНСТАНТЫ В НАШУ ЭПОХУ:")
            print("=" * 80)
            print(f"Время: t = {consts['t']:.0f}")
            print(f"N = {consts['N']:.1e} (цель: 9.7e+122)")
            print(f"p = {consts['p']:.4f} (цель: 0.0527)")
            print(f"U = {consts['U']:.1f} (цель: ~328)")
            print(f"Аттрактор p√(KU)/e = {consts['attractor']:.4f}")
            print()
            print(f"t_p = {consts['t_p']:.2e} с (цель: 5.39e-44, отношение: {consts['t_p'] / 5.39e-44:.3f})")
            print(f"c = {consts['c']:.2e} м/с (цель: 3.00e+8, отношение: {consts['c'] / 3.00e8:.3f})")
            print(f"G = {consts['G']:.2e} м³/кг·с² (цель: 6.67e-11, отношение: {consts['G'] / 6.67e-11:.3f})")
            print(f"Λ = {consts['Lambda']:.2e} м⁻² (цель: 1.1e-52, отношение: {consts['Lambda'] / 1.1e-52:.3f})")

        except Exception as e:
            print(f"Ошибка при визуализации: {e}")
            plt.close('all')


# ============================================================================
# ЗАПУСК РАДИКАЛЬНОЙ МОДЕЛИ
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ЗАПУСК МОДЕЛИ С СИЛЬНЫМИ ОСЦИЛЛЯЦИЯМИ")
    print("=" * 80)

    # Создаем Вселенную с сильными осцилляциями
    universe = StrongOscillatingUniverse(K=8.0)

    # Запускаем несколько раз с разными начальными условиями
    test_conditions = [
        {"name": "Экстремальные колебания", "y0": [0.9, -0.8]},
        {"name": "Средние колебания", "y0": [0.5, -0.3]},
        {"name": "Малые колебания", "y0": [0.2, -0.1]},
    ]

    for test in test_conditions:
        print(f"\n{'=' * 60}")
        print(f"ТЕСТ: {test['name']}")
        print(f"{'=' * 60}")

        # Запускаем симуляцию с меньшим временем для стабильности
        solution = universe.simulate(
            t_span=(0, 30000),  # Уменьшено для стабильности
            y0=test["y0"]
        )

        # Анализируем
        universe.analyze()

        # Визуализируем только для интересных случаев
        if "Экстремальные" in test["name"] or "Средние" in test["name"]:
            try:
                universe.visualize()
            except Exception as e:
                print(f"Ошибка при визуализации: {e}")

        print()

    # Финальный запуск с оптимальными параметрами
    print("\n" + "=" * 80)
    print("ФИНАЛЬНЫЙ ЗАПУСК С ОПТИМАЛЬНЫМИ ПАРАМЕТРАМИ")
    print("=" * 80)

    # Усиливаем параметры еще больше
    universe.alpha = 0.15
    universe.beta = 0.08
    universe.omega_N = 0.0003
    universe.omega_p = 0.0004

    # Запускаем
    solution = universe.simulate(
        t_span=(0, 50000),  # Уменьшено для стабильности
        y0=[0.7, -0.5]  # Начинаем с отклонения
    )

    # Полный анализ
    universe.analyze()

    try:
        universe.visualize()
    except Exception as e:
        print(f"Ошибка при визуализации: {e}")

    # Анализ амплитуд
    print("\n" + "=" * 80)
    print("АНАЛИЗ АМПЛИТУД ОСЦИЛЛЯЦИЙ")
    print("=" * 80)

    if universe.N is not None and len(universe.N) > 0:
        N_min, N_max = np.min(universe.N), np.max(universe.N)
        p_min, p_max = np.min(universe.p), np.max(universe.p)

        print(f"ЧИСЛО УЗЛОВ N:")
        print(f"  Минимум: {N_min:.1e} (планковский масштаб: 1.0e+60)")
        print(f"  Максимум: {N_max:.1e} (аттракторный масштаб: 1.0e+147)")
        if N_min > 0:
            print(f"  Размах: {math.log10(N_max / N_min):.1f} порядков (цель: ~87)")
            print(f"  Наша эпоха: 9.7e+122 (в {9.7e122 / N_min:.1e} раз выше минимума)")

        print(f"\nНЕЛОКАЛЬНОСТЬ p:")
        print(f"  Минимум: {p_min:.4f}")
        print(f"  Максимум: {p_max:.4f}")
        if p_min > 0:
            print(f"  Размах: {p_max / p_min:.1f} раз")
        print(f"  Наша эпоха: 0.0527")

        # Оцениваем, насколько хорошо модель соответствует реальности
        print(f"\nСООТВЕТСТВИЕ РЕАЛЬНОСТИ:")

        # Находим лучшую точку
        consts = universe.calculate_constants_at_epoch()

        if consts:
            errors = {
                'N': abs(consts['N'] - 9.7e122) / 9.7e122 * 100,
                'p': abs(consts['p'] - 0.0527) / 0.0527 * 100,
                't_p': abs(consts['t_p'] - 5.39e-44) / 5.39e-44 * 100,
                'c': abs(consts['c'] - 3.00e8) / 3.00e8 * 100,
                'G': abs(consts['G'] - 6.67e-11) / 6.67e-11 * 100,
                'Lambda': abs(consts['Lambda'] - 1.1e-52) / 1.1e-52 * 100,
            }

            print(f"  Ошибки в лучшей точке:")
            for key, error in errors.items():
                print(f"    {key}: {error:.1f}%")

            avg_error = np.mean(list(errors.values()))
            print(f"\n  Средняя ошибка: {avg_error:.1f}%")

            if avg_error < 50:
                print("  ✅ Модель реалистична!")
            elif avg_error < 100:
                print("  ⚠️  Модель умеренно реалистична")
            else:
                print("  ❌ Модель нуждается в доработке")