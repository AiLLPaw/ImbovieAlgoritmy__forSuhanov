import tensorflow as tf
import string
import random
import time

# Проверяем доступность GPU для TensorFlow
# (На Windows для полноценного CUDA рекомендуется использовать WSL2)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[INFO] TensorFlow успешно использует GPU: {gpus[0].name}")
    device_name = '/GPU:0'
else:
    print("[INFO] Внимание: TensorFlow работает на CPU (для GPU на Windows используйте WSL2).")
    device_name = '/CPU:0'


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
    Генерирует матрицу заданного размера:
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
    Если выбранная точка является тупиком (0), находит ближайшую доступную ячейку.
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
                        print(f"[УСПЕХ] Найдена замена: {index_to_col_letter(nc)}{nr + 1}")
                        return nr, nc
                    queue.append((nr, nc))

    raise ValueError("Не удалось найти ни одной проходимой ячейки во всей матрице!")


def wave_algorithm_tf_smart(matrix, start, finish):
    """
    Волновой поиск пути с расчетом контрольной суммы и подсчетом обогнутых тупиков.
    """
    with tf.device(device_name):
        grid = tf.constant(matrix, dtype=tf.float32)
        rows = tf.shape(grid)[0].numpy()
        cols = tf.shape(grid)[1].numpy()

        sr, sc = start
        fr, fc = finish

        # Авто-коррекция старта и финиша при попадании в тупик
        sr, sc = find_nearest_valid_cell(matrix, sr, sc, rows, cols)
        fr, fc = find_nearest_valid_cell(matrix, fr, fc, rows, cols)

        print("[INFO] Запуск волнового распространения...")
        start_time = time.time()

        distances = tf.fill((rows, cols), float('inf'))
        indices = tf.constant([[sr, sc]], dtype=tf.int32)
        updates = tf.constant([0.0], dtype=tf.float32)
        distances = tf.tensor_scatter_nd_update(distances, indices, updates)

        valid_mask = grid > 0.0

        max_iterations = rows * cols
        for i in range(max_iterations):
            old_distances = tf.identity(distances)

            up = tf.concat([tf.fill((1, cols), float('inf')), distances[:-1, :]], axis=0)
            down = tf.concat([distances[1:, :], tf.fill((1, cols), float('inf'))], axis=0)
            left = tf.concat([tf.fill((rows, 1), float('inf')), distances[:, :-1]], axis=1)
            right = tf.concat([distances[:, 1:], tf.fill((rows, 1), float('inf'))], axis=1)

            best_neighbor = tf.minimum(tf.minimum(up, down), tf.minimum(left, right))
            new_dist = best_neighbor + grid

            update_condition = (new_dist < distances) & valid_mask
            distances = tf.where(update_condition, new_dist, distances)

            if tf.reduce_all(distances == old_distances):
                break

        print(f"[INFO] Расчеты завершены за {time.time() - start_time:.4f} сек.")

        final_dist = distances[fr, fc].numpy()
        if final_dist == float('inf'):
            return None, 0.0, 0

        # Восстановление пути
        dist_np = distances.numpy()
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

        # 1. Расчет контрольной суммы весов пути (гарантированно >= 0)
        total_cost = 0.0
        for r, c in path:
            cell_val = grid_np[r][c]
            if cell_val < 0.0:
                cell_val = 0.0
            total_cost += cell_val
        total_cost = max(0.0, total_cost)

        # 2. Подсчет уникальных тупиков (нулей), которые алгоритм обогнул вблизи пути
        bypassed_obstacles = set()
        path_set = set(path)
        for r, c in path:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid_np[nr][nc] == 0.0 and (nr, nc) not in path_set:
                        bypassed_obstacles.add((nr, nc))

        return path, total_cost, len(bypassed_obstacles)


if __name__ == "__main__":
    print("=== УМНЫЙ ВОЛНОВОЙ МАРШРУТИЗАТОР ===")

    # --- НАСТРОЙКА ОТОБРАЖЕНИЯ ---
    # True  — обрезать длинный список пути (показать первые и последние 10 шагов)
    # False — выводить список пути ПОЛНОСТЬЮ целиком (удобно для скриптов визуализации)
    TRUNCATE_PATH = False

    NUM_ROWS = int(input("Введите количество строк (например, 20): "))
    NUM_COLS = int(input("Введите количество столбцов (например, 20): "))

    max_col_letter = index_to_col_letter(NUM_COLS - 1)
    print(f"-> Сетка {NUM_ROWS}x{NUM_COLS}. Столбцы: A — {max_col_letter}")

    matrix_data = generate_matrix_with_obstacle_coeff(NUM_ROWS, NUM_COLS, obstacle_chance=0.15, max_weight=15.0)

    print("\n--- Ввод координат ---")
    start_col_input = input(f"Буква столбца старта (от A до {max_col_letter}): ").strip()
    start_row_input = int(input(f"Номер строки старта (от 1 до {NUM_ROWS}): "))

    finish_col_input = input(f"Буква столбца финиша (от A до {max_col_letter}): ").strip()
    finish_row_input = int(input(f"Номер строки финиша (от 1 до {NUM_ROWS}): "))

    start_pos = (start_row_input - 1, col_letter_to_index(start_col_input))
    finish_pos = (finish_row_input - 1, col_letter_to_index(finish_col_input))

    try:
        path, checksum, obstacles_count = wave_algorithm_tf_smart(matrix_data, start_pos, finish_pos)
        if path:
            print(f"\n[УСПЕХ] Путь успешно построен!")
            print(f"• Общее количество шагов: {len(path)}")
            print(f"• Контрольная сумма весов пути: {checksum:.2f} (защита от < 0 активна)")
            print(f"• Обогнуто уникальных тупиков (нулей) рядом с путем: {obstacles_count}")

            formatted_path = [f"{index_to_col_letter(c)}{r + 1}" for r, c in path]

            print("\nСписок шагов пути:")
            if TRUNCATE_PATH and len(formatted_path) > 20:
                print("Первые шаги:", " -> ".join(formatted_path[:10]))
                print("...")
                print("Последние шаги:", " -> ".join(formatted_path[-10:]))
            else:
                # Полный вывод для визуализации
                print(" -> ".join(formatted_path))
        else:
            print("\n[ОТКАЗ] Путь полностью заблокирован тупиками.")
    except Exception as e:
        print(f"\n[ОШИБКА] {e}")


