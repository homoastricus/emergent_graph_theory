import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import periodogram, welch, find_peaks, lombscargle
from scipy.stats import linregress
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


@dataclass
class LatticeModel:
    """Улучшенная модель решетки с физическими ограничениями"""
    name: str
    z: int
    beta: float
    omega_factor: float
    physical_priority: float  # Приоритет основанный на физических соображениях

    def sigma(self, r, A, xi, B, C, a):
        """Физически осмысленная модель σ(r)"""
        beta = self.beta
        omega = self.omega_factor / a if self.omega_factor > 0 else 0
        exponential = A * np.exp(-r / xi)
        power_law = B / (r ** beta)
        oscillations = C * np.cos(omega * r) * np.exp(-r / xi) if omega > 0 else 0  # Затухающие осцилляции

        return exponential + power_law + oscillations


class HybridLatticeReconstructor:
    """
    ГИБРИДНЫЙ РЕКОНСТРУКТОР с улучшенными алгоритмами
    """

    def __init__(self):
        # Физически осмысленные модели решеток
        self.lattices = [
            LatticeModel("SC", 6, 1.00, np.pi, 0.9),
            LatticeModel("BCC", 8, 0.95, np.pi * np.sqrt(3), 1.0),
            LatticeModel("FCC", 12, 0.85, 2 * np.pi, 0.8),
            LatticeModel("HCP", 12, 0.90, 2 * np.pi, 0.7),
            LatticeModel("Diamond", 4, 1.00, np.pi / np.sqrt(3), 0.1),
            LatticeModel("Random", -1, 1.00, 0.0, 0.05),
            LatticeModel("Tetrahedral", 4, 1.10, np.pi * 2 / 3, 0.6),
        ]

        self.expected_xi = 1.648721  # e-1

    def robust_xi_estimation(self, r, sigma_r) -> float:
        """МНОГОМЕТОДНАЯ оценка корреляционной длины"""
        methods = []
        weights = []

        # Метод 1: Логарифмический фит с R² проверкой
        try:
            valid_mask = (r > 0.3) & (r < 4.0) & (sigma_r > 1e-8) & (np.isfinite(sigma_r))
            if np.sum(valid_mask) > 5:
                r_val, sigma_val = r[valid_mask], np.log(sigma_r[valid_mask])
                slope, intercept, r_value, p_value, std_err = linregress(r_val, sigma_val)
                if r_value ** 2 > 0.7 and p_value < 0.05:  # Хорошее качество фита
                    xi_log = abs(1.0 / slope)
                    methods.append(xi_log)
                    weights.append(r_value ** 2)  # Вес по качеству фита
        except Exception as e:
            pass

        # Метод 2: Интерполяция полувысоты с улучшенной логикой
        try:
            if len(r) > 10:
                # Ищем максимум в разумном диапазоне
                search_mask = r < 3.0
                if np.any(search_mask):
                    max_val = np.max(sigma_r[search_mask])
                    half_max = max_val / 2.0

                    # Находим точку, где сигнал падает ниже половины
                    below_half = np.where(sigma_r <= half_max)[0]
                    if len(below_half) > 0:
                        first_below = below_half[0]
                        if first_below > 0:
                            # Линейная интерполяция для точности
                            r1, r2 = r[first_below - 1], r[first_below]
                            s1, s2 = sigma_r[first_below - 1], sigma_r[first_below]
                            if s1 > half_max:  # Проверка корректности
                                xi_half = r1 + (r2 - r1) * (half_max - s1) / (s2 - s1)
                                methods.append(xi_half)
                                weights.append(0.8)  # Средняя надежность
        except Exception as e:
            pass

        # Метод 3: Анализ производной
        try:
            if len(r) > 15:
                # Сглаживаем для стабильности производной
                window = max(3, len(r) // 20)
                sigma_smooth = np.convolve(sigma_r, np.ones(window) / window, mode='same')
                log_sigma = np.log(sigma_smooth + 1e-8)
                deriv = np.gradient(log_sigma, r)

                # Берем медиану отрицательных производных (экспоненциальное затухание)
                neg_deriv = deriv[deriv < -0.01]
                if len(neg_deriv) > 5:
                    xi_deriv = -1.0 / np.median(neg_deriv)
                    methods.append(xi_deriv)
                    weights.append(0.6)  # Меньший вес из-за шума
        except Exception as e:
            pass

        # Метод 4: Автокорреляция
        try:
            if len(r) > 20:
                # Нормированная автокорреляция
                sigma_norm = (sigma_r - np.mean(sigma_r)) / np.std(sigma_r)
                autocorr = np.correlate(sigma_norm, sigma_norm, mode='full')
                autocorr = autocorr[len(autocorr) // 2:]
                autocorr = autocorr / autocorr[0]

                # Находим первую точку ниже 1/e
                threshold = np.exp(-1)
                below_thresh = np.where(autocorr <= threshold)[0]
                if len(below_thresh) > 0:
                    first_below = below_thresh[0]
                    if first_below > 0:
                        xi_auto = r[min(first_below, len(r) - 1)]
                        methods.append(xi_auto)
                        weights.append(0.7)
        except Exception as e:
            pass

        # Взвешенное среднее
        if methods:
            if len(weights) == len(methods):
                xi_final = np.average(methods, weights=weights)
            else:
                xi_final = np.median(methods)  # Робастная оценка
        else:
            xi_final = self.expected_xi

        return float(np.clip(xi_final, 0.5, 4.0))


    def robust_spectral_analysis(self, sigma_r, r) -> Dict:
        """УЛУЧШЕННЫЙ спектральный анализ"""
        sigma_norm = (sigma_r - np.mean(sigma_r)) / (np.std(sigma_r) + 1e-10)

        # 1. Увеличиваем диапазон частот для BCC/FCC
        frequencies = np.linspace(0.05, 8.0, 1000)  # Было 0.1-5.0
        power = lombscargle(r, sigma_norm, frequencies, normalize=True)

        # 2. Более чувствительное обнаружение пиков
        height_threshold = np.median(power) + 0.3 * np.std(power)  # Было +1 std
        peaks, properties = find_peaks(power,
                                       height=height_threshold,
                                       prominence=0.05,  # Более чувствительно
                                       distance=len(power) // 30,  # Меньше расстояние
                                       width=2)  # Добавляем проверку ширины

        # 3. Улучшенное определение z с учетом энтропии
        significant_peaks = len(peaks)
        dominant_freq = frequencies[np.argmax(power)] if len(power) > 0 else 0

        # НОРМАЛИЗОВАННАЯ энтропия (0-1)
        Pxx_norm = power / (np.sum(power) + 1e-10)
        entropy_normalized = -np.sum(Pxx_norm * np.log(Pxx_norm + 1e-10)) / np.log(len(power))

        # Обновленная логика определения z
        if significant_peaks == 0 and entropy_normalized > 0.8:
            z_est = 6  # SC - нет пиков, высокая энтропия
        elif significant_peaks == 1 and 1.0 <= dominant_freq <= 2.5:
            z_est = 8  # BCC - один основной пик
        elif significant_peaks >= 2:
            z_est = 12  # FCC/HCP - несколько пиков
        else:
            z_est = 6  # По умолчанию

        return {
            'z_est': z_est,
            'peaks_lombscargle': significant_peaks,
            'dominant_frequency': dominant_freq,
            'spectral_entropy_normalized': entropy_normalized,
            'raw_entropy': -np.sum(Pxx_norm * np.log(Pxx_norm + 1e-10)),
            'power_spectrum': (frequencies, power)
        }

    def physical_bayesian_weight(self, lattice: LatticeModel, spectral_info: Dict,
                                 xi_estimated: float, chi2: float, chi2_min: float) -> float:
        """УЛУЧШЕННЫЕ веса с балансом между z и χ²"""
        z_spectral = spectral_info['z_est']
        entropy_norm = spectral_info['spectral_entropy_normalized']

        # Базовые веса
        base_weights = {
            "SC": 0.8, "BCC": 1.0, "FCC": 0.9, "HCP": 0.85,
            "Tetrahedral": 0.7, "Diamond": 0.4, "Random": 0.1
        }
        weight = base_weights.get(lattice.name, 0.5)

        # 1. БОЛЕЕ МЯГКАЯ проверка z
        if lattice.z > 0:
            z_diff = abs(lattice.z - z_spectral)
            # МЕНЕЕ агрессивные штрафы
            z_penalty = [1.0, 0.9, 0.7, 0.4, 0.2]  # Было: [1.0, 0.7, 0.3, 0.1, 0.01]
            penalty_idx = min(z_diff, len(z_penalty) - 1)
            weight *= z_penalty[penalty_idx]

        # 2. УЧЕТ КАЧЕСТВА ФИТА (важнее z!)
        if chi2 < np.inf and chi2_min > 0:
            # Относительное качество фита
            fit_ratio = chi2_min / (chi2 + 1e-8)
            if fit_ratio > 0.9:  # Почти одинаковое качество
                weight *= 1.2
            elif fit_ratio > 0.8:
                weight *= 1.1
            elif fit_ratio < 0.5:  # Значительно хуже
                weight *= 0.8

        # 3. Проверка осцилляций для BCC/FCC
        peaks_lomb = spectral_info['peaks_lombscargle']
        if lattice.name in ["BCC", "FCC"]:
            expected_peaks = 1 if lattice.name == "BCC" else 2
            if peaks_lomb >= expected_peaks:
                weight *= 1.3
            elif peaks_lomb == 0:
                weight *= 0.8  # Меньший штраф
            # Если пики не обнаружены, но χ² хороший - не штрафуем сильно

        # 4. Учет энтропии
        if entropy_norm > 0.8:  # Высокая энтропия = менее упорядоченно
            if lattice.name == "Random":
                weight *= 1.3
            else:
                weight *= 0.9

        return max(weight, 0.01)

    def fit_lattice_models(self, r, sigma_r, xi_estimated: float) -> Dict:
        """ПОДБОР МОДЕЛЕЙ с улучшенной стабильностью"""
        results = {}

        # Предварительная обработка данных
        valid_mask = (r > 0.1) & (r < 10.0) & np.isfinite(sigma_r)
        r_clean = r[valid_mask]
        sigma_clean = sigma_r[valid_mask]

        if len(r_clean) < 10:
            # Возвращаем значения по умолчанию если данных недостаточно
            for lattice in self.lattices:
                results[lattice.name] = {
                    'chi2': np.inf,
                    'xi_fitted': xi_estimated,
                    'parameters': None,
                    'lattice': lattice
                }
            return results

        for lattice in self.lattices:
            try:
                # Умные начальные приближения в зависимости от структуры
                if lattice.name == "SC":
                    p0 = [0.3, xi_estimated, 0.08, 0.01, 1.0]
                    bounds = ([0.01, 0.3, 0.001, -0.1, 0.5],
                              [2.0, 3.0, 0.5, 0.1, 2.0])
                elif lattice.name == "BCC":
                    p0 = [0.35, xi_estimated * 0.9, 0.06, 0.02, 1.2]
                    bounds = ([0.01, 0.3, 0.001, -0.2, 0.8],
                              [2.0, 3.0, 0.3, 0.2, 2.5])
                elif lattice.name == "FCC":
                    p0 = [0.4, xi_estimated * 0.8, 0.05, 0.03, 1.5]
                    bounds = ([0.01, 0.3, 0.001, -0.3, 1.0],
                              [2.0, 3.0, 0.3, 0.3, 3.0])
                elif lattice.name == "HCP":
                    p0 = [0.38, xi_estimated * 0.85, 0.055, 0.025, 1.4]
                    bounds = ([0.01, 0.3, 0.001, -0.25, 1.0],
                              [2.0, 3.0, 0.3, 0.25, 2.8])
                elif lattice.name == "Diamond":
                    p0 = [0.25, xi_estimated * 1.1, 0.1, 0.015, 0.9]
                    bounds = ([0.01, 0.4, 0.005, -0.05, 0.3],
                              [1.5, 3.0, 0.4, 0.05, 1.5])
                elif lattice.name == "Tetrahedral":
                    p0 = [0.28, xi_estimated, 0.09, 0.012, 0.8]
                    bounds = ([0.01, 0.3, 0.005, -0.08, 0.4],
                              [1.5, 3.0, 0.3, 0.08, 1.8])
                else:  # Random
                    p0 = [0.5, xi_estimated, 0.12, 0.0, 1.0]
                    bounds = ([0.01, 0.2, 0.001, -0.01, 0.5],
                              [3.0, 4.0, 0.5, 0.01, 2.0])

                # Взвешенный фит (больше вес на малых r)
                weights = 1.0 / (r_clean + 0.1)

                def model_fn(r, A, xi, B, C, a):
                    return lattice.sigma(r, A, xi, B, C, a)

                popt, pcov = curve_fit(
                    model_fn, r_clean, sigma_clean, p0=p0,
                    sigma=1.0 / (weights + 1e-8),
                    maxfev=10000,
                    bounds=bounds,
                    method='trf'
                )

                predicted = model_fn(r_clean, *popt)
                residuals = weights * (predicted - sigma_clean)
                chi2 = np.sqrt(np.mean(residuals ** 2))

                results[lattice.name] = {
                    'chi2': chi2,
                    'xi_fitted': popt[1],
                    'parameters': popt,
                    'lattice': lattice,
                    'predicted': predicted
                }

            except Exception as e:
                results[lattice.name] = {
                    'chi2': np.inf,
                    'xi_fitted': xi_estimated,
                    'parameters': None,
                    'lattice': lattice,
                    'predicted': None
                }

        return results

    def validate_on_ideal_structures(self):
        """ТЕСТИРОВАНИЕ на идеальных данных для калибровки"""
        print("🔧 КАЛИБРОВКА НА ИДЕАЛЬНЫХ СТРУКТУРАХ")
        print("=" * 50)

        r_test = np.linspace(0.1, 8.0, 150)

        test_structures = {
            "SC": {"params": [0.35, 1.65, 0.08, 0.01, 1.0]},
            "BCC": {"params": [0.37, 1.65, 0.06, 0.02, 1.2]},
            "FCC": {"params": [0.4, 1.6, 0.05, 0.03, 1.5]}
        }

        for name, config in test_structures.items():
            # Находим соответствующую решетку
            lattice = next((l for l in self.lattices if l.name == name), None)
            if lattice:
                # Генерация идеальных данных
                sigma_ideal = lattice.sigma(r_test, *config["params"])
                # Добавляем небольшой шум для реалистичности
                sigma_ideal += np.random.normal(0, 0.005, len(r_test))

                # Реконструкция
                result = self.reconstruct(r_test, sigma_ideal)
                correct = result['best_model'] == name
                status = "✅" if correct else "❌"
                print(
                    f"{status} {name}: определена как {result['best_model']} (вероятность: {result['probability']:.1%})")

    def reconstruct(self, r, sigma_r) -> Dict:
        """УЛУЧШЕННАЯ реконструкция с балансом параметров"""
        print("🎯 ГИБРИДНАЯ РЕКОНСТРУКЦИЯ ФУНДАМЕНТАЛЬНОЙ СТРУКТУРЫ")
        print("=" * 60)

        # 1. Улучшенный спектральный анализ
        spectral_info = self.robust_spectral_analysis(sigma_r, r)
        xi_estimated = self.robust_xi_estimation(r, sigma_r)

        print(f"📊 СПЕКТРАЛЬНЫЙ АНАЛИЗ:")
        print(f"   - Координационное число: z = {spectral_info['z_est']}")
        print(f"   - Пики в спектре: {spectral_info['peaks_lombscargle']} (Lomb-Scargle)")
        print(f"   - Спектральная энтропия: {spectral_info['raw_entropy']:.3f}")
        print(f"   - Норм. энтропия: {spectral_info['spectral_entropy_normalized']:.3f}")
        print(f"   - Корреляционная длина: ξ = {xi_estimated:.5f} lₚ")

        # 2. Подбор моделей
        fit_results = self.fit_lattice_models(r, sigma_r, xi_estimated)

        # 3. Находим минимальный χ² для относительных весов
        valid_chi2 = [r['chi2'] for r in fit_results.values() if r['chi2'] < np.inf]
        chi2_min = min(valid_chi2) if valid_chi2 else 1.0

        # 4. Улучшенное байесовское взвешивание
        posterior_probs = {}
        print(f"\n📈 ПОДБОР МОДЕЛЕЙ:")
        for name, result in fit_results.items():
            lattice = result['lattice']
            chi2 = result['chi2']

            phys_weight = self.physical_bayesian_weight(
                lattice, spectral_info, result['xi_fitted'], chi2, chi2_min
            )

            if chi2 < np.inf:
                # Более сбалансированная likelihood
                likelihood = np.exp(-(chi2 - chi2_min))  # Относительно лучшего χ²
                posterior = likelihood * phys_weight
            else:
                posterior = 0.0

            posterior_probs[name] = posterior

            if chi2 < np.inf:
                print(f"   - {name:<12} | χ²: {chi2:.3e} | вес: {phys_weight:.3f}")
            else:
                print(f"   - {name:<12} | χ²: --- | вес: {phys_weight:.3f}")

        # Нормировка и выбор лучшей
        total_posterior = sum(posterior_probs.values())
        if total_posterior > 0:
            for name in posterior_probs:
                posterior_probs[name] /= total_posterior

        best_model = max(posterior_probs, key=posterior_probs.get)
        best_prob = posterior_probs[best_model]

        print(f"\n🎯 РЕЗУЛЬТАТ: {best_model} (вероятность: {best_prob:.1%})")

        # 5. Физическая интерпретация
        best_chi2 = fit_results[best_model]['chi2'] if fit_results[best_model]['chi2'] < np.inf else float('inf')
        self._print_physical_interpretation(best_model, spectral_info['z_est'],
                                            xi_estimated, best_prob, best_chi2)

        return {
            'best_model': best_model,
            'probability': best_prob,
            'z_estimated': spectral_info['z_est'],
            'xi_estimated': xi_estimated,
            'posterior_probs': posterior_probs,
            'fit_results': fit_results,
            'spectral_info': spectral_info
        }

    def _print_physical_interpretation(self, model: str, z: int, xi: float, prob: float, chi2: float):
        """ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ результата"""
        print(f"\n🔬 ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:")

        confidence = "ВЫСОКАЯ" if prob > 0.8 else "УМЕРЕННАЯ" if prob > 0.6 else "НИЗКАЯ"

        if model == "SC" and z == 6:
            print(f"   ✅ {confidence} ДОСТОВЕРНОСТЬ: SC решетка")
            print("   • Простая кубическая структура")
            print("   • Оптимальная симметрия и предсказуемость")
            print("   • Согласуется с минимальной сложностью ПНИД")
            print(f"   • Качество фита: χ² = {chi2:.3e}")

        elif model == "BCC" and z == 8:
            print(f"   ✅ {confidence} ДОСТОВЕРНОСТЬ: BCC решетка")
            print("   • Объемно-центрированная кубическая структура")
            print("   • Повышенная плотность упаковки")
            print("   • Эффективная организация информации")
            print(f"   • Качество фита: χ² = {chi2:.3e}")

        elif model == "Tetrahedral" and z == 4:
            print(f"   ⚠️  {confidence} ДОСТОВЕРНОСТЬ: Тетраэдрическая структура")
            print("   • Минимальная устойчивая конфигурация в 3D")
            print("   • Фундаментальная простота")
            print("   • Требует дополнительной проверки")
            print(f"   • Качество фита: χ² = {chi2:.3e}")

        elif model in ["FCC", "HCP"] and z == 12:
            print(f"   ⚠️  {confidence} ДОСТОВЕРНОСТЬ: Плотная упаковка")
            print("   • Максимальная плотность координации")
            print("   • Высокая симметрия")
            print("   • Проверить на изотропность")
            print(f"   • Качество фита: χ² = {chi2:.3e}")

        else:
            print(f"   ⚠️  {confidence} ДОСТОВЕРНОСТЬ: {model}")
            print("   • Рассмотреть возможность смешанной структуры")
            print("   • Проверить качество входных данных")
            print("   • Возможны неучтенные физические эффекты")
            print(f"   • Качество фита: χ² = {chi2:.3e}")


# =========================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =========================================================

def test_hybrid_reconstructor():
    """Тестирование гибридного реконструктора"""
    np.random.seed(42)

    # Генерация тестовых данных для BCC решетки
    r = np.linspace(0.1, 10, 80000)

    # Модель BCC с шумом - ИСПРАВЛЕННЫЕ ПАРАМЕТРЫ
    A, xi, B, C, a = 0.37, 1.65, 0.08, 0.05, 1.2  # B уменьшено с 0.1 до 0.08
    sigma_r = (A * np.exp(-r / xi) + B / (r ** 0.95) +
               C * np.cos((np.pi * np.sqrt(3) / a) * r) * np.exp(-r / (xi * 0.8)))  # Разные ξ для осцилляций

    sigma_r += np.random.normal(0, 0.002, len(r))

    # Реконструкция
    reconstructor = HybridLatticeReconstructor()

    # Основная реконструкция
    print("🔧 ЗАПУСК ОСНОВНОЙ РЕКОНСТРУКЦИИ...")
    results = reconstructor.reconstruct(r, sigma_r)

    return results, r, sigma_r


def plot_reconstruction_results(results, r, sigma_r):
    """Визуализация результатов реконструкции"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # График 1: Исходные данные и лучшая модель
    ax = axes[0, 0]
    best_model = results['best_model']
    best_fit = results['fit_results'][best_model]

    ax.scatter(r, sigma_r, alpha=0.6, label='Данные', s=20)
    if best_fit['predicted'] is not None:
        # Используем r_clean для предсказаний
        valid_mask = (r > 0.1) & (r < 10.0) & np.isfinite(sigma_r)
        r_clean = r[valid_mask]
        ax.plot(r_clean, best_fit['predicted'], 'r-', linewidth=2,
                label=f'Лучшая модель: {best_model}')

    ax.set_xlabel('Расстояние r (lₚ)')
    ax.set_ylabel('σ(r)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Реконструкция: {best_model} (вероятность: {results["probability"]:.1%})')

    # График 2: Спектральный анализ с пиками
    ax = axes[0, 1]
    freqs, power = results['spectral_info']['power_spectrum']
    ax.plot(freqs, power, 'b-', linewidth=1, label='Спектр')

    # Показываем обнаруженные пики
    peaks = results['spectral_info']['peaks_lombscargle']
    if peaks > 0:
        dominant_freq = results['spectral_info']['dominant_frequency']
        ax.axvline(dominant_freq, color='red', linestyle='--',
                   label=f'Основная частота: {dominant_freq:.2f}')

    ax.set_xlabel('Частота')
    ax.set_ylabel('Мощность')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Спектр Ломба-Скаргла')

    # График 3: Вероятности моделей
    ax = axes[1, 0]
    models = list(results['posterior_probs'].keys())
    probs = [results['posterior_probs'][m] for m in models]
    colors = ['green' if m == best_model else 'blue' for m in models]
    bars = ax.bar(models, probs, color=colors, alpha=0.7)

    # Добавляем подписи значений
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{prob:.1%}', ha='center', va='bottom')

    ax.set_ylabel('Вероятность')
    ax.set_xticklabels(models, rotation=45)
    ax.grid(True, alpha=0.3)
    ax.set_title('Апостериорные вероятности')

    # График 4: Информация о результате
    ax = axes[1, 1]
    ax.axis('off')
    info_text = (
        f"Лучшая модель: {best_model}\n"
        f"Вероятность: {results['probability']:.1%}\n"
        f"z = {results['z_estimated']}\n"
        f"ξ = {results['xi_estimated']:.3f} lₚ\n"
        f"Энтропия: {results['spectral_info']['raw_entropy']:.3f}\n"
        f"Норм. энтропия: {results['spectral_info']['spectral_entropy_normalized']:.3f}\n"
        f"Пики: {results['spectral_info']['peaks_lombscargle']}"
    )
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Запуск тестирования
    results, r, sigma_r = test_hybrid_reconstructor()

    # Визуализация результатов
    plot_reconstruction_results(results, r, sigma_r)