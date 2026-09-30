# -*- coding: utf-8 -*-
"""Модуль математического кодирования и генерации многоракурсных лентикулярных растров."""

import cv2
import numpy as np


class LenticularGenerator(object):
    """Класс, инкапсулирующий генерацию серии 3D-ракурсов по карте глубин

    и попиксельную сборку (интерлейсинг) финальной матрицы для лентикулярной печати.
    """

    def __init__(self, params, state_obj, anaglyph_processor):
        self.lpi = params["lpi"]
        self.dpi = params["dpi"]
        self.print_w_inch = params["print_w_inch"]
        self.views_count = params["views_count"]
        self.max_shift = params["max_shift"]
        self.src_img = params["src_image"]
        
        self.state = state_obj
        self.ap = anaglyph_processor

    def run(self, progress_callback):
        """Запускает оптимизированный по памяти цикл генерации лентикулярного мастера."""
        # 1. Расчет геометрии мастер-кадра для печати
        h_src, w_src, _ = self.src_img.shape
        aspect_ratio = float(w_src) / float(h_src)
        
        total_print_w_px = int(self.print_w_inch * self.dpi)
        total_print_h_px = int(total_print_w_px / aspect_ratio)

        progress_callback(0.05, "Extracting high-resolution depth map...")

        # Формирование карты глубин (с учетом настроек микрорельефа)
        if self.state.use_microrelief:
            depth_map = self.ap.dp.get_detailed_depth(self.state.blur, self.state.opacity)
        else:
            depth_map = self.ap.dp.raw_depth.copy()
        
        # Масштабирование исходных данных до целевого разрешения печати (CUBIC для фото)
        master_img = cv2.resize(self.src_img, (total_print_w_px, total_print_h_px), interpolation=cv2.INTER_CUBIC)
        master_depth = cv2.resize(depth_map, (total_print_w_px, total_print_h_px), interpolation=cv2.INTER_LINEAR)
        depth_normalized = master_depth.astype(np.float32) / 255.0

        if self.state.depth_interpretation == "logarithmic":
            depth_normalized = (np.exp(depth_normalized * 3.0) - 1.0) / (np.exp(3.0) - 1.0)
            depth_normalized = np.clip(depth_normalized, 0.0, 1.0)

        # Опорные координатные сетки
        x_indices, y_indices = np.meshgrid(np.arange(total_print_w_px), np.arange(total_print_h_px))
        
        # Создаем пустой холст под финальный отпечаток (занимает память один раз)
        lenticular_canvas = np.zeros_like(master_img)
        
        # Физический размер одной линзы в пикселях принтера
        lens_width_px = float(self.dpi) / self.lpi

        # === ПРЕДРАСЧЕТ КАРТЫ СООТВЕТСТВИЯ СТОЛБЦОВ И РАКУРСОВ ===
        # Заранее вычисляем, какой столбец X какому индексу ракурса принадлежит
        x_to_view_map = np.zeros(total_print_w_px, dtype=np.int32)
        for x in range(total_print_w_px):
            sub_lens_pos = (float(x) / lens_width_px) % 1.0
            view_target_idx = int((1.0 - sub_lens_pos) * self.views_count)
            x_to_view_map[x] = max(0, min(view_target_idx, self.views_count - 1))

        # === СТРИМИНГОВЫЙ ПОТОКОВЫЙ КОНВЕЙЕР (ОПТИМИЗАЦИЯ RAM ПОД OOM-KILLER) ===
        # Генерируем по одному ракурсу за итерацию, вырезаем маску столбцов и сразу удаляем ракурс из RAM
        for v_idx in range(self.views_count):
            progress_callback(
                0.10 + 0.85 * (float(v_idx) / float(self.views_count)), 
                f"Processing stream phase {v_idx + 1} of {self.views_count}..."
            )
            
            # Находим все столбцы X, которые должны взять пиксели из этого ракурса
            columns_mask = (x_to_view_map == v_idx)
            if not np.any(columns_mask):
                continue  # Если под линзы этот ракурс нигде не попал, пропускаем тяжелый ремап

            # Коэффициент смещения по фазе: от -0.5 до +0.5
            t = (float(v_idx) / float(self.views_count - 1)) - 0.5
            shift_factor = depth_normalized * (self.max_shift * t)
            
            map_x = (x_indices + shift_factor).astype(np.float32)
            map_y = y_indices.astype(np.float32)
            
            # Генерируем ОДИН временный ракурс
            view_frame = cv2.remap(master_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            
            # Переносим столбцы ракурса на итоговый холст по векторной маске NumPy (очень быстро)
            lenticular_canvas[:, columns_mask] = view_frame[:, columns_mask]
            
            # Явно освобождаем локальные тяжелые массивы текущего шага для сборщика мусора (GC)
            del view_frame
            del map_x
            del map_y
            del shift_factor

        return lenticular_canvas

