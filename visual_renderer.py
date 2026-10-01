# -*- coding: utf-8 -*-
"""Модуль графического рендеринга интерфейса: математика холста и косинусный фезернинг."""

import os
import time
import cv2
import numpy as np


class VisualRenderer(object):
    """Класс, отвечающий за компоновку финального кадра, масштабирование,

    отрисовку заблюренного фона с градиентным уходом в черный цвет, OSD и Help.
    """

    def __init__(self, state_obj, anaglyph_processor):
        self.state = state_obj  # Инжектируем стейт строго в поле self.state
        self.ap = anaglyph_processor

        # 1. Кеш для горизонтального градиента фона (черные края экрана)
        self._cached_gradient_mask = None
        self._cached_bg_geometry_key = None  # Ключ: (scr_w, scr_h, start_x, new_w)

        # 2. Кеш для карты косинусного размытия стыка кадра (feather_mask)
        self._cached_feather_mask = None
        self._cached_fg_geometry_key = None  # Ключ: (scr_w, scr_h, start_x, start_y, new_w, new_h)

        # === СЧИТЫВАНИЕ СПРАВКИ ИЗ ВНЕШНЕГО ФАЙЛА ПРИ ИНИЦИАЛИЗАЦИИ ===
        self.help_lines = []
        help_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "help.txt")
        
        try:
            if os.path.exists(help_path):
                with open(help_path, "r", encoding="utf-8") as f:
                    self.help_lines = [line.rstrip() for line in f.readlines()]
            else:
                raise FileNotFoundError
        except Exception:
            # Защитная заглушка на случай утери файла на диске
            self.help_lines = [
                "=== INTERACTIVE VIEWER HELP ===",
                "[-] Error: 'help.txt' file not found on disk.",
                "Please recreate help.txt in application directory."
            ]

    def get_fit_to_screen_scale(self, img_w, img_h):
        """Расчет коэффициента для вписывания картинки в экран."""
        return min(self.state.screen_w / img_w, self.state.screen_h / img_h)

    def _generate_fast_gradient_mask(self, scr_w, scr_h, start_x, new_w):
        """Векторизованная генерация маски градиента фона без использования циклов."""
        x_coords = np.arange(scr_w, dtype=np.float32)
        gradient_line = np.ones(scr_w, dtype=np.float32)

        img_left_edge = start_x
        img_right_edge = start_x + new_w

        if img_left_edge > 0:
            left_mask = x_coords < img_left_edge
            gradient_line[left_mask] = x_coords[left_mask] / img_left_edge

        denom = scr_w - img_right_edge
        if denom > 0:
            right_mask = x_coords >= img_right_edge
            gradient_line[right_mask] = 1.0 - ((x_coords[right_mask] - img_right_edge) / denom)

        # Векторный Smoothstep
        gradient_line = 3 * (gradient_line ** 2) - 2 * (gradient_line ** 3)
        gradient_mask = np.tile(gradient_line, (scr_h, 1))
        return np.atleast_3d(gradient_mask)

    def _generate_feather_mask(self, scr_w, scr_h, c_x1, c_y1, c_x2, c_y2):
        """Вычисляет карту расстояний и строит косинусную маску сглаживания стыка кадра."""
        photo_mask = np.zeros((scr_h, scr_w), dtype=np.uint8)
        if (c_x2 > c_x1) and (c_y2 > c_y1):
            photo_mask[c_y1:c_y2, c_x1:c_x2] = 255

        # Расчет расстояний Distance Transform
        dist_transform = cv2.distanceTransform(photo_mask, cv2.DIST_L2, 5)

        # Косинусный фезернинг шириной в 41 пиксель
        feather_size = 41.0
        linear_mask = np.clip(dist_transform / feather_size, 0.0, 1.0)
        feather_mask = 0.5 - 0.5 * np.cos(linear_mask * np.pi)
        return np.atleast_3d(feather_mask.astype(np.float32))

    def _render_wiggle_frame(self, processed_img):
        """Генерирует монокулярный сдвиг кадра по синусоиде времени для эффекта Wiggle 3D."""
        if not self.state.has_depth:
            return processed_img.copy()
            
        img_h, img_w, _ = processed_img.shape
        depth_map = self.ap.dp.get_detailed_depth(self.state.blur, self.state.opacity) if self.state.use_microrelief else self.ap.dp.raw_depth
        if depth_map.shape[:2] != (img_w, img_h):
            depth_map = cv2.resize(depth_map, (img_w, img_h))
            
        depth_normalized = depth_map.astype(np.float32) / 255.0
        shift_amplitude = np.sin(self.state.wiggle_phase * 2.0 * np.pi)
        shift_factor = depth_normalized * (self.state.disparity * 0.5 * shift_amplitude)
        
        x_indices, y_indices = np.meshgrid(np.arange(img_w), np.arange(img_h))
        map_x = (x_indices + shift_factor).astype(np.float32)
        map_y = y_indices.astype(np.float32)
        return cv2.remap(processed_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def render_canvas(self, processed_img, current_filename, total_count, file_index, anaglyph_proc):
        """Формирует финальный холст с оптимизированным заблюренным фоном и HUD."""
        self.ap = anaglyph_proc
        scr_w, scr_h = self.state.screen_w, self.state.screen_h

        # Если активирован Wiggle 3D, пересчитываем параллакс кадра на лету
        target_img = processed_img
        if self.state.view_mode == "wiggle":
            target_img = self._render_wiggle_frame(self.ap.dp.primary_img)

        img_h, img_w, _ = target_img.shape
        new_w = max(int(img_w * self.state.scale), 1)
        new_h = max(int(img_h * self.state.scale), 1)
        resized = cv2.resize(target_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Валидация смещений кадрирования
        if new_w > scr_w:
            self.state.offset_x = int(np.clip(self.state.offset_x, -(new_w - scr_w) // 2, (new_w - scr_w) // 2))
        else:
            self.state.offset_x = 0

        if new_h > scr_h:
            self.state.offset_y = int(np.clip(self.state.offset_y, -(new_h - scr_h) // 2, (new_h - scr_h) // 2))
        else:
            self.state.offset_y = 0

        # === АППАРАТНАЯ СВЕРХБЫСТРАЯ ГЕНЕРАЦИЯ БОКЕ ЗАДНЕГО ПЛАНА ===
        ds_factor = 4
        ds_w, ds_h = max(scr_w // ds_factor, 1), max(scr_h // ds_factor, 1)
        bg_downscaled = cv2.resize(target_img, (ds_w, ds_h), interpolation=cv2.INTER_LINEAR)
        
        blur_kernel_dim = int(scr_h * 0.023)
        if blur_kernel_dim % 2 == 0: blur_kernel_dim += 1
        blur_kernel_dim = max(3, blur_kernel_dim)
        
        bg_blurred_low = cv2.GaussianBlur(bg_downscaled, (blur_kernel_dim, blur_kernel_dim), 0)
        canvas_base = cv2.resize(bg_blurred_low, (scr_w, scr_h), interpolation=cv2.INTER_LINEAR)

        # Расчет координат наложения основного кадра
        start_x = (scr_w - new_w) // 2 + self.state.offset_x
        start_y = (scr_h - new_h) // 2 + self.state.offset_y

        # КЕШИРОВАНИЕ МАСКИ ЗАДНЕГО ПЛАНА (УХОД В ЧЕРНЫЙ К КРАЯМ)
        bg_geometry_key = (scr_w, scr_h, start_x, new_w)
        if self._cached_gradient_mask is None or self._cached_bg_geometry_key != bg_geometry_key:
            self._cached_gradient_mask = self._generate_fast_gradient_mask(scr_w, scr_h, start_x, new_w)
            self._cached_bg_geometry_key = bg_geometry_key
        
        canvas_bg = (canvas_base.astype(np.float32) * self._cached_gradient_mask).astype(np.uint8)

        # Подготовка слоев наложения
        canvas_fg = canvas_bg.copy()
        c_x1, c_y1 = max(0, start_x), max(0, start_y)
        c_x2, c_y2 = min(scr_w, start_x + new_w), min(scr_h, start_y + new_h)
        r_x1, r_y1 = max(0, -start_x), max(0, -start_y)
        r_x2, r_y2 = r_x1 + (c_x2 - c_x1), r_y1 + (c_y2 - c_y1)

        if (c_x2 > c_x1) and (c_y2 > c_y1):
            canvas_fg[c_y1:c_y2, c_x1:c_x2] = resized[r_y1:r_y2, r_x1:r_x2]

        # КЕШИРОВАНИЕ МАСКИ СТЫКОВКИ КАДРОВ ЧЕРЕЗ DISTANCE TRANSFORM
        fg_geometry_key = (scr_w, scr_h, start_x, start_y, new_w, new_h)
        if self._cached_feather_mask is None or self._cached_fg_geometry_key != fg_geometry_key:
            self._cached_feather_mask = self._generate_feather_mask(scr_w, scr_h, c_x1, c_y1, c_x2, c_y2)
            self._cached_fg_geometry_key = fg_geometry_key

        # ПОЛНОЭКРАННЫЙ МАТЕМАТИЧЕСКИЙ АЛЬФА-БЛЕНДИНГ СЛОЕВ
        canvas = (canvas_fg.astype(np.float32) * self._cached_feather_mask + 
                  canvas_bg.astype(np.float32) * (1.0 - self._cached_feather_mask)).astype(np.uint8)

        # === ВЫЧИСЛЕНИЕ АДАПТИВНОЙ ПРОЗРАЧНОСТИ HUD (ТОЛЬКО В РЕЖИМЕ СЛАЙД-ШОУ) ===
        idle_time = time.time() - self.state.last_input_time
        
        # Если слайд-шоу запущено, панели плавно тают через 2.5 секунды неактивности.
        # Если слайд-шоу на паузе, HUD всегда горит со 100% яркостью (hud_alpha = 1.0).
        if self.state.slideshow_active:
            hud_alpha = max(0.0, 1.0 - (idle_time - 2.5) / 0.3) if idle_time > 2.5 else 1.0
        else:
            hud_alpha = 1.0

        # Наложение HUD-интерфейсов с учетом вычисленного коэффициента альфы
        if self.state.show_osd and hud_alpha > 0:
            osd_layer = self._draw_osd(canvas.copy(), current_filename, total_count, file_index)
            cv2.addWeighted(osd_layer, hud_alpha, canvas, 1.0 - hud_alpha, 0, canvas)
            
        if self.state.show_help and hud_alpha > 0:
            help_layer = self._draw_help(canvas.copy())
            cv2.addWeighted(help_layer, hud_alpha, canvas, 1.0 - hud_alpha, 0, canvas)
            
        if self.state.slideshow_active:
            # Виджет таймера рендерится и затухает только при активном слайд-шоу
            timer_layer = self._draw_slideshow_timer(canvas.copy())
            cv2.addWeighted(timer_layer, hud_alpha, canvas, 1.0 - hud_alpha, 0, canvas)

        return canvas


    # === ПРОДОЛЖЕНИЕ КЛАССА VisualRenderer: ВСТРОЕННАЯ ГИСТОГРАММА И ИНТЕРФЕЙС ===

    def _draw_osd(self, canvas, current_filename, total_count, file_index):
        """Внутренний метод отрисовки OSD-сайдбара со встроенной гистограммой планов сцены."""
        scr_h = self.state.screen_h
        m_name = self.ap.method_details.get(self.state.method_name, self.state.method_name)
        orig_h, orig_w, _ = self.ap.dp.primary_img.shape
        
        mode_status = self.state.view_mode.upper() if not self.state.fb_3d_mode else "FACEBOOK 3D"
        order_label = "SEQ" if self.state.slideshow_order == "sequential" else "RND"
        ss_status = f"ACTIVE ({order_label} | {self.state.slideshow_delay:.1f}s)" if self.state.slideshow_active else "OFF"

        osd_lines = [
            f"FILE: {current_filename} [{file_index + 1}/{total_count}]",
            f"RESOLUTION: {orig_w} x {orig_h} px",
            f"VIEW MODE: {mode_status} (F2: WIGGLE)",
            f"ANAGLYPH METHOD: {m_name if (self.state.view_mode == 'anaglyph' and self.state.has_depth) else 'N/A'}",
            f"GEOMETRY: {self.state.depth_interpretation.upper() if self.state.has_depth else 'N/A'}",
            f"MICRORELIEF: {'ON' if (self.state.use_microrelief and self.state.has_depth) else 'OFF'}",
            f"DISPARITY (STEREO BASE): {self.state.disparity if self.state.has_depth else '0'} (8/*)",
            f"BLUR RADIUS: {self.state.blur if self.state.has_depth else 'N/A'}",
            f"OPACITY STRENGTH: {f'{self.state.opacity:.2f}' if self.state.has_depth else 'N/A'}",
            f"CONVERGENCE SHIFT: {self.state.convergence if self.state.has_depth else 'N/A'} (5/6)",
            f"SCALE: {self.state.scale * 100:.1f}% (0/1/2)",
            f"SLIDESHOW PLAYBACK: {ss_status}",
            f"GHOST REDUCTION (F8): {'ON' if self.state.anaglyph_ghost_reduction else 'OFF'}",
            f"WIGGLE FREQ: {self.state.wiggle_speed:.1f} Hz ([ / ])" if self.state.view_mode == "wiggle" else ""
        ]
        
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (520, scr_h), (0, 0, 0), -1)

        # --- ИНТЕГРАЦИЯ: ВСТРОЕННАЯ ГИСТОГРАММА ПЛАНОВ КАРТЫ ГЛУБИН ---
        if self.state.has_depth:
            # Геометрия окна гистограммы внутри OSD-панели (смещаем в нижнюю треть сайдбара)
            hist_w, hist_h = 256, 100
            bx, by = 25, scr_h - hist_h - 40
            
            # Извлекаем текущую карту глубин
            depth = self.ap.dp.get_detailed_depth(self.state.blur, self.state.opacity) if self.state.use_microrelief else self.ap.dp.raw_depth
            hist_data = cv2.calcHist([depth], [0], None, [256], [0, 256])
            cv2.normalize(hist_data, hist_data, 0, hist_h - 15, cv2.NORM_MINMAX)
            
            # Контур рамки гистограммы внутри сайдбара
            cv2.rectangle(overlay, (bx, by), (bx + hist_w, by + hist_h), (20, 20, 20), -1)
            cv2.rectangle(overlay, (bx, by), (bx + hist_w, by + hist_h), (100, 100, 100), 1)
            
            # Отрисовываем вертикальные спектральные столбики
            for i in range(256):
                val = int(hist_data[i])
                if val > 0:
                    # Хроматический переход: от дальнего плана (синий) к ближнему плану (красный)
                    color = (i, 255 - i, 50 + i // 2)
                    cv2.line(overlay, (bx + i, by + hist_h - 1), (bx + i, by + hist_h - val), color, 1)
            
            # Текстовые подписи планов внутри рамки
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(overlay, "DEPTH GRAPH (OSD):", (bx, by - 10), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay, "BG", (bx + 5, by + 15), font, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(overlay, "FG", (bx + hist_w - 25, by + 15), font, 0.32, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        # Отрисовка текстовых строк поверх сайдбара
        y_pos = 45
        for line in osd_lines:
            if line:
                cv2.putText(canvas, line, (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
                y_pos += 30
        return canvas

    def _draw_help(self, canvas):
        """Внутренний метод отрисовки сайдбара справки у правого края."""
        scr_w, scr_h = self.state.screen_w, self.state.screen_h
        box_w = 580
        start_x = scr_w - box_w

        overlay = canvas.copy()
        cv2.rectangle(overlay, (start_x, 0), (scr_w, scr_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

        y_pos = 45
        for line in self.help_lines:
            color = (0, 255, 255) if (line.startswith("=== ") or line.startswith("---")) else (255, 255, 255)
            cv2.putText(canvas, line, (start_x + 25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
            y_pos += 26
        return canvas

    def _draw_slideshow_timer(self, canvas):
        """Отрисовывает аккуратный плавающий виджет обратного отсчета в левом нижнем углу."""
        scr_h = self.state.screen_h
        remaining_secs = max(0.0, self.state.slideshow_delay - self.state.slideshow_accumulator)
        timer_text = f"NEXT IN: {remaining_secs:.1f}s"
        
        box_w, box_h = 180, 40
        start_x = 20
        start_y = scr_h - box_h - 20
        
        overlay = canvas.copy()
        cv2.rectangle(overlay, (start_x, start_y), (start_x + box_w, start_y + box_h), (0, 0, 0), -1)
        border_color = (0, 0, 255) if remaining_secs <= 1.0 else (0, 255, 0)
        cv2.rectangle(overlay, (start_x, start_y), (start_x + box_w, start_y + box_h), border_color, 1)
        cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0, canvas)
        
        cv2.putText(canvas, timer_text, (start_x + 15, start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

