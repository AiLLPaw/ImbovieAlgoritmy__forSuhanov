import torch
import string
import random
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Проверяем доступность CUDA (GPU) для PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"[INFO] PyTorch успешно использует GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[INFO] Внимание: PyTorch работает на CPU.")


def col_letter_to_index(letter: str) -> int:
    """Конвертирует букву столбца (например, 'A' -> 0) в индекс."""
    letter = letter.upper()
    idx = 0
    for char in letter:
        idx = idx * 26 + (ord(char) - ord('A') + 1)
    return idx - 1


def index_to_col_letter(idx: int) -> str:
    """Конвертирует числовой индекс обратно в букву столбца."""
    result = []
    idx += 1
    while idx > 0:
        idx, remainder = divmod(idx - 1, 26)
        result.append(string.ascii_uppercase[remainder])
    return "".join(reversed(result))


def generate_matrix_with_obstacle_coeff(rows, cols, obstacle_chance=0.15, max_weight=15.0):
    """
    Генерирует матрицу весов:
    0.0 — тупик (препятствие)
    >0.0 — вес ячейки (поддерживает значения больше 2)
    """
    print(f"[INFO] Генерация матрицы {rows}x{cols} (коэффициент тупиков: {obstacle_chance * 100}%)...")
    matrix = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if random.random() < obstacle_chance:
                row.append(0.0)
            else:
                row.append(round(random.uniform(1.0, max_weight), 1))
        matrix.append(row)
    return matrix


def find_nearest_valid_cell(matrix, r, c, rows, cols):
    """
    Если выбранная точка — тупик (0), ищет ближайшую доступную ячейку.
    """
    if matrix[r][c] > 0.0:
        return r, c

    print(f"[ПРЕДУПРЕЖДЕНИЕ] Точка {index_to_col_letter(c)}{r + 1} — тупик (0). Ищем ближайшую замену...")
    queue = [(r, c)]
    visited = {(r, c)}

    while queue:
        curr_r, curr_c = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    if matrix[nr][nc] > 0.0:
                        return nr, nc
                    queue.append((nr, nc))

    raise ValueError("Не удалось найти ни одной проходимой ячейки во всей матрице!")


def wave_algorithm_torch(matrix, start, finish):
    """
    Волновой поиск пути на PyTorch (CUDA) с расчетами на тензорах.
    """
    grid = torch.tensor(matrix, dtype=torch.float32, device=device)
    rows, cols = grid.shape

    sr, sc = start
    fr, fc = finish

    # Авто-коррекция старта и финиша
    sr, sc = find_nearest_valid_cell(matrix, sr, sc, rows, cols)
    fr, fc = find_nearest_valid_cell(matrix, fr, fc, rows, cols)

    print("[INFO] Запуск волнового распространения на GPU (PyTorch)...")
    start_time = time.time()

    # Инициализация матрицы расстояний (бесконечность везде)
    distances = torch.full((rows, cols), float('inf'), device=device)
    distances[sr, sc] = 0.0

    valid_mask = grid > 0.0

    # Итеративное распространение волны (минимизация путей по соседям)
    max_iterations = rows * cols
    for i in range(max_iterations):
        old_distances = distances.clone()

        # Сдвиги соседей в PyTorch (через torch.roll)
        up = torch.roll(distances, shifts=1, dims=0)
        down = torch.roll(distances, shifts=-1, dims=0)
        left = torch.roll(distances, shifts=1, dims=1)
        right = torch.roll(distances, shifts=-1, dims=1)

        # Обрезаем переходы через границы (края матрицы)
        up[0, :] = float('inf')
        down[-1, :] = float('inf')
        left[:, 0] = float('inf')
        right[:, -1] = float('inf')

        best_neighbor = torch.min(torch.min(up, down), torch.min(left, right))
        new_dist = best_neighbor + grid  # Прибавляем вес ячейки

        # Обновляем расстояния
        mask = (new_dist < distances) & valid_mask
        distances = torch.where(mask, new_dist, distances)

        if torch.equal(distances, old_distances):
            break

    print(f"[INFO] Расчеты завершены за {time.time() - start_time:.4f} сек.")

    if distances[fr, fc].item() == float('inf'):
        return None, 0.0, 0, (sr, sc), (fr, fc)

    # Переводим матрицу расстояний обратно в numpy для восстановления пути
    dist_np = distances.cpu().numpy()
    grid_np = matrix

    path = [(fr, fc)]
    curr_r, curr_c = fr, fc

    while (curr_r, curr_c) != (sr, sc):
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid_np[nr][nc] > 0.0:
                neighbors.append((dist_np[nr, nc], nr, nc))

        if not neighbors:
            break
        _, curr_r, curr_c = min(neighbors, key=lambda x: x[0])
        path.append((curr_r, curr_c))

    path.reverse()

    # Расчет контрольной суммы весов пути (строго >= 0)
    total_cost = sum(max(0.0, grid_np[r][c]) for r, c in path)

    # Подсчет уникальных тупиков (нулей), которые обогнул алгоритм
    bypassed_obstacles = set()
    path_set = set(path)
    for r, c in path:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, curr_c + dc  # исправлено в уме, проверяем окрестность r+dr, c+dc
            # точнее:
    # (перепишем корректный подсчет ниже)
    bypassed_obstacles = set()
    for r, c in path:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid_np[nr][nc] == 0.0 and (nr, nc) not in path_set:
                    bypassed_obstacles.add((nr, nc))

    return path, total_cost, len(bypassed_obstacles), (sr, sc), (fr, fc)


def plot_grid_and_path_green(matrix, path, start, finish, checksum):
    """
    Визуализация с зеленой градацией весов, обведенными тупиками
    и диаграммой роста контрольной суммы.
    """
    grid_data = np.array(matrix, dtype=float)
    rows, cols = grid_data.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.3, 1]})

    # --- 1. КАРТА С ЗЕЛЕНОЙ ГРАДАЦИЕЙ И ОБВОДКОЙ ТУПИКОВ (Слева) ---
    # Используем палитру оттенков зеленого ('Greens') для контраста с черным
    cmap = plt.cm.Greens.copy()

    masked_grid = np.ma.masked_where(grid_data == 0, grid_data)
    im = ax1.imshow(masked_grid, cmap=cmap, origin='upper', vmin=0.1)

    # Тупики заливаем черным цветом
    obstacle_mask = (grid_data == 0)
    ax1.imshow(np.where(obstacle_mask, 1, np.nan), cmap=plt.cm.Greys, origin='upper', vmin=0, vmax=1)

    # Сетка для четкого разделения и обводки ячеек
    ax1.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax1.grid(which="minor", color="#555555", linestyle="-", linewidth=0.6, alpha=0.5)
    ax1.tick_params(which="minor", size=0)

    # Отрисовка пути (красная линия с оранжевыми узлами)
    if path:
        path_cols = [c for r, c in path]
        path_rows = [r for r, c in path]

        ax1.plot(path_cols, path_rows, color='red', linewidth=2.5, label='Путь', zorder=4)
        ax1.scatter(path_cols, path_rows, color='orange', s=12, zorder=5)

        ax1.scatter(start[1], start[0], color='lime', s=160, edgecolors='black',
                    linewidth=1.5, label=f'Старт ({index_to_col_letter(start[1])}{start[0] + 1})', zorder=6)
        ax1.scatter(finish[1], finish[0], color='deepskyblue', marker='*', s=260, edgecolors='black',
                    linewidth=1.5, label=f'Финиш ({index_to_col_letter(finish[1])}{finish[0] + 1})', zorder=6)

    col_labels = [index_to_col_letter(i) for i in range(cols)]
    row_labels = [str(i + 1) for i in range(rows)]
    step_c = max(1, cols // 15)
    step_r = max(1, rows // 15)

    ax1.set_xticks(ticks=range(0, cols, step_c))
    ax1.set_xticklabels(col_labels[::step_c])
    ax1.set_yticks(ticks=range(0, rows, step_r))
    ax1.set_yticklabels(row_labels[::step_r])

    ax1.set_title(f"Волновой маршрут (PyTorch / CUDA)\nШагов: {len(path)} | Итог вес: {checksum:.2f}", fontsize=11)
    ax1.set_xlabel("Столбцы")
    ax1.set_ylabel("Строки")
    ax1.legend(loc='upper right', fontsize=9)

    # --- 2. ДИАГРАММА РОСТА КОНТРОЛЬНОЙ СУММЫ (Справа) ---
    if path:
        cumulative_costs = []
        current_sum = 0.0
        for r, c in path:
            val = max(0.0, grid_data[r][c])
            current_sum += val
            cumulative_costs.append(current_sum)

        step_indices = list(range(1, len(path) + 1))

        ax2.plot(step_indices, cumulative_costs, color='forestgreen', linewidth=2.5, marker='o', markersize=3)
        ax2.fill_between(step_indices, cumulative_costs, color='forestgreen', alpha=0.15)

        ax2.set_title("Диаграмма накопления контрольной суммы пути", fontsize=11)
        ax2.set_xlabel("Номер шага по порядку", fontsize=10)
        ax2.set_ylabel("Накопленный суммарный вес", fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)

        ax2.annotate(f'Итог: {checksum:.2f}',
                     xy=(step_indices[-1], cumulative_costs[-1]),
                     xytext=(step_indices[-1] * 0.7, cumulative_costs[-1] * 0.85),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                     fontsize=10, fontweight='bold', color='darkgreen')

    plt.tight_layout()
    print("\n[INFO] Открытие графического окна...")
    plt.show()


if __name__ == "__main__":
    print("=== УМНЫЙ МАРШРУТИЗАТОР (PyTorch + CUDA) ===")

    TRUNCATE_PATH = False  # True — обрезать вывод, False — полный вывод для визуализации

    NUM_ROWS = int(input("Введите количество строк (например, 20): "))
    NUM_COLS = int(input("Введите количество столбцов (например, 20): "))

    max_col_letter = index_to_col_letter(NUM_COLS - 1)
    matrix_data = generate_matrix_with_obstacle_coeff(NUM_ROWS, NUM_COLS, obstacle_chance=0.15, max_weight=15.0)

    print("\n--- Ввод координат ---")
    start_col_input = input(f"Буква столбца старта (от A до {max_col_letter}): ").strip()
    start_row_input = int(input(f"Номер строки старта (от 1 до {NUM_ROWS}): "))

    finish_col_input = input(f"Буква столбца финиша (от A до {max_col_letter}): ").strip()
    finish_row_input = int(input(f"Номер строки финиша (от 1 до {NUM_ROWS}): "))

    start_pos = (start_row_input - 1, col_letter_to_index(start_col_input))
    finish_pos = (finish_row_input - 1, col_letter_to_index(finish_col_input))

    try:
        path, checksum, obstacles_count, actual_start, actual_finish = wave_algorithm_torch(matrix_data, start_pos,
                                                                                            finish_pos)

        if path:
            print(f"\n[УСПЕХ] Путь построен!")
            print(f"• Шагов: {len(path)}")
            print(f"• Контрольная сумма весов: {checksum:.2f} (гарантированно >= 0)")
            print(f"• Обогнуто тупиков: {obstacles_count}")

            formatted_path = [f"{index_to_col_letter(c)}{r + 1}" for r, c in path]
            print("\nСписок шагов пути:")
            if TRUNCATE_PATH and len(formatted_path) > 20:
                print("Первые:", " -> ".join(formatted_path[:10]))
                print("...")
                print("Последние:", " -> ".join(formatted_path[-10:]))
            else:
                print(" -> ".join(formatted_path))

            # Запуск графического окна
            plot_grid_and_path_green(matrix_data, path, actual_start, actual_finish, checksum)
        else:
            print("\n[ОТКАЗ] Путь полностью заблокирован.")
    except Exception as e:
        print(f"\n[ОШИБКА] {e}")