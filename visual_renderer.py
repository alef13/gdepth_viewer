# -*- coding: utf-8 -*-
"""Модуль графического рендеринга интерфейса: математика холста и косинусный фезернинг."""

import cv2
import numpy as np


class VisualRenderer(object):
    """Класс, отвечающий за компоновку финального кадра, масштабирование

    и отрисовку заблюренного фона с градиентным уходом в черный цвет.
    """

    def __init__(self, state_obj, anaglyph_processor):
        self.state = state_obj  # Инжектируем стейт строго в поле self.state
        self.ap = anaglyph_processor

    def get_fit_to_screen_scale(self, img_w, img_h):
        """Расчет коэффициента для вписывания картинки в экран."""
        return min(self.state.screen_w / img_w, self.state.screen_h / img_h)

    def render_canvas(self, processed_img, current_filename, total_count, file_index, anaglyph_proc):
        """Формирует финальный холст с заблюренным градиентным фоном и HUD."""
        self.ap = anaglyph_proc

        img_h, img_w, _ = processed_img.shape
        scr_w, scr_h = self.state.screen_w, self.state.screen_h

        # 1. Расчет размеров целевого (вписанного/отмасштабированного) изображения
        new_w = max(int(img_w * self.state.scale), 1)
        new_h = max(int(img_h * self.state.scale), 1)
        resized = cv2.resize(processed_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 2. Валидация смещений кадрирования
        if new_w > scr_w:
            max_offset_x = (new_w - scr_w) // 2
            self.state.offset_x = int(np.clip(self.state.offset_x, -max_offset_x, max_offset_x))
        else:
            self.state.offset_x = 0

        if new_h > scr_h:
            max_offset_y = (new_h - scr_h) // 2
            self.state.offset_y = int(np.clip(self.state.offset_y, -max_offset_y, max_offset_y))
        else:
            self.state.offset_y = 0

        # === ГЕНЕРАЦИЯ ЗАБЛЮРЕННОГО ЗАДНЕГО ПЛАНА ===
        bg_fill = cv2.resize(processed_img, (scr_w, scr_h), interpolation=cv2.INTER_LINEAR)
        canvas_base = cv2.GaussianBlur(bg_fill, (99, 99), 0)

        # 3. Расчет координат наложения основного кадра
        start_x = (scr_w - new_w) // 2 + self.state.offset_x
        start_y = (scr_h - new_h) // 2 + self.state.offset_y

        # === СОЗДАНИЕ ПЛАВНОГО ГОРИЗОНТАЛЬНОГО ГРАДИЕНТА В ЧЕРНЫЙ ===
        gradient_line = np.zeros(scr_w, dtype=np.float32)
        img_left_edge = start_x
        img_right_edge = start_x + new_w

        for x in range(scr_w):
            if x < img_left_edge:
                gradient_line[x] = float(x) / img_left_edge if img_left_edge > 0 else 1.0
            elif x >= img_right_edge:
                denom = scr_w - img_right_edge
                gradient_line[x] = 1.0 - (float(x - img_right_edge) / denom) if denom > 0 else 1.0
            else:
                gradient_line[x] = 1.0

        # S-образное сглаживание фона к краям экрана
        gradient_line = 3 * (gradient_line ** 2) - 2 * (gradient_line ** 3)
        gradient_mask = np.tile(gradient_line, (scr_h, 1))
        gradient_mask = np.atleast_3d(gradient_mask)
        
        # Получаем готовый затухающий фон
        canvas_bg = (canvas_base.astype(np.float32) * gradient_mask).astype(np.uint8)

        # === СОЗДАНИЕ ВЫДЕЛЕННОГО ПОЛНОЭКРАННОГО СЛОЯ С ОРИГИНАЛЬНОЙ КАРТИНКОЙ ===
        canvas_fg = canvas_bg.copy()
        c_x1, c_y1 = max(0, start_x), max(0, start_y)
        c_x2, c_y2 = min(scr_w, start_x + new_w), min(scr_h, start_y + new_h)
        r_x1, r_y1 = max(0, -start_x), max(0, -start_y)
        r_x2, r_y2 = r_x1 + (c_x2 - c_x1), r_y1 + (c_y2 - c_y1)

        if (c_x2 > c_x1) and (c_y2 > c_y1):
            canvas_fg[c_y1:c_y2, c_x1:c_x2] = resized[r_y1:r_y2, r_x1:r_x2]

        # === ГЕНЕРАЦИЯ АЛЬФА-МАСКИ ЧЕРЕЗ DISTANCE TRANSFORM И КОСИНУСНУЮ ИНТЕРПОЛЯЦИЮ ===
        photo_mask = np.zeros((scr_h, scr_w), dtype=np.uint8)
        photo_mask[c_y1:c_y2, c_x1:c_x2] = 255

        # Точное евклидово расстояние от пикселей внутри фото до его границ
        dist_transform = cv2.distanceTransform(photo_mask, cv2.DIST_L2, 5)

        # Желаемая ширина мягкого косинусного перехода в пикселях
        feather_size = 11.0
        
        # Линейная нормализация коридора (0.0 на краю, 1.0 в глубине кадра)
        linear_mask = np.clip(dist_transform / feather_size, 0.0, 1.0)
        
        # Превращаем линейный спад в косинусный (Smoothstep / Cosine Fade)
        # Формула: 0.5 - 0.5 * cos(linear_mask * pi)
        # На краях (где 0.0) даст 0.5 - 0.5*1 = 0.0. В глубине (где 1.0) даст 0.5 - 0.5*(-1) = 1.0.
        feather_mask = 0.5 - 0.5 * np.cos(linear_mask * np.pi)
        feather_mask = np.atleast_3d(feather_mask.astype(np.float32))

        # === ПОЛНОЭКРАННЫЙ МАТЕМАТИЧЕСКИЙ АЛЬФА-БЛЕНДИНГ ===
        canvas = (canvas_fg.astype(np.float32) * feather_mask + 
                  canvas_bg.astype(np.float32) * (1.0 - feather_mask)).astype(np.uint8)

        # Наложение HUD-интерфейсов (без изменений во втором блоке)
        if self.state.show_osd:
            canvas = self._draw_osd(canvas, current_filename, total_count, file_index)
        if self.state.show_help:
            canvas = self._draw_help(canvas)
        if self.state.slideshow_active:
            canvas = self._draw_slideshow_timer(canvas)

        return canvas
    
    # === ПРОДОЛЖЕНИЕ КЛАССА VisualRenderer: ОТРИСОВКА ИНТЕРФЕЙСА СТАТУСОВ И ПОМОЩИ ===

    def _draw_osd(self, canvas, current_filename, total_count, file_index):
        """Внутренний метод отрисовки OSD-сайдбара."""
        scr_h = self.state.screen_h
        m_name = self.ap.method_details.get(self.state.method_name, self.state.method_name)
        
        if not self.state.has_depth:
            mode_status = "2D FLAT IMAGE (NO 3D METADATA)"
        else:
            mode_status = "FACEBOOK 3D LOOK" if self.state.fb_3d_mode else self.state.view_mode.upper()
            
        order_label = "SEQ" if self.state.slideshow_order == "sequential" else "RND"
        ss_status = f"ACTIVE ({order_label} | {self.state.slideshow_delay:.1f}s)" if self.state.slideshow_active else "OFF"

        osd_lines = [
            f"FILE: {current_filename} [{file_index + 1}/{total_count}]",
            f"VIEW MODE: {mode_status}",
            f"ANAGLYPH METHOD: {m_name if (self.state.view_mode == 'anaglyph' and self.state.has_depth) else 'N/A'}",
            f"GEOMETRY: {self.state.depth_interpretation.upper() if self.state.has_depth else 'N/A'}",
            f"MICRORELIEF: {'ON' if (self.state.use_microrelief and self.state.has_depth) else 'OFF'}",
            f"DISPARITY (STEREO BASE): {self.state.disparity if self.state.has_depth else '0'}",
            f"BLUR RADIUS: {self.state.blur if self.state.has_depth else 'N/A'}",
            f"OPACITY STRENGTH: {f'{self.state.opacity:.2f}' if self.state.has_depth else 'N/A'}",
            f"CONVERGENCE SHIFT: {self.state.convergence if self.state.has_depth else 'N/A'}",
            f"SCALE: {self.state.scale * 100:.1f}%",
            f"OFFSET: {self.state.offset_x}, {self.state.offset_y}",
            f"SLIDESHOW PLAYBACK: {ss_status}"
        ]
        
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (520, scr_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        y_pos = 45
        for line in osd_lines:
            cv2.putText(canvas, line, (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
            y_pos += 30
        return canvas

    def _draw_help(self, canvas):
        """Внутренний метод отрисовки сайдбара справки у правого края."""
        scr_w, scr_h = self.state.screen_w, self.state.screen_h
        help_lines = [
            "=== INTERACTIVE VIEWER HELP ===",
            "[ ESC ]             Exit viewport session",
            "[ SPACE ]           Toggle Slideshow (Play / Pause)",
            "[ TAB ]             Toggle Left-Side OSD Panel",
            "[ F1 ]              Toggle this Help Window",
            "[ F2 ]              Export frame to PNG on disk",
            "[ F3 ]              Switch mode to Original Flat 2D Photo",
            "[ F4 ]              Switch mode to horizontal Stereopair",
            "[ F5 ]              Switch mode to Depth Map Jet Color",
            "[ F6 ]              Toggle Geometry (LINEAR / LOG)",
            "[ F7 ]              Toggle ВЧ-Microrelief ON / OFF",
            "[ F8 ]              Export Pydantic state backup dump",
            "[ F9 ]              Toggle Slideshow autoplay ON / OFF",
            "[ F10 ]             Toggle Slideshow Order (SEQ / SHUFFLE)",
            "[ F11 ]             Toggle Interactive Facebook 3D Look (Mouse Mode)",
            "--- RED-CYAN ANAGLYPH MATRICES ---",
            "[ Q W E R T Y U I O P ] -> 10 basic empirical methods",
            "--- ADVANCED MATRICES ---",
            "[ A S K L ]         -> Dubois, Olson, McAllister, Rossman",
            "[ D ]               -> Spectrum Optimized Olson Method",
            "[ Z ]               -> Green-Magenta Empirical Pure Matrix",
            "[ X ]               -> Green-Magenta Dubois Linear Method",
            "--- 3D MODIFIERS & CANVAS CONTROLS ---",
            "[ / ] / [ * ]       -> Scale Stereo Base down / up",
            "[ INS ] / [ DEL ]   -> Shift Parallax Plane up / down",
            " /       -> Scale Detail Filter down / up",
            " /       -> Scale Detail Strength down / up",
            "[ + ] / [ - ]       -> Canvas Zoom In / Zoom Out step 5%",
            " /       -> Reset scale / Auto-Fit to Screen",
            "[ ARROW KEYS ]      -> Pan oversized image matrix",
        ]
        
        box_w = 580
        start_x = scr_w - box_w

        overlay = canvas.copy()
        cv2.rectangle(overlay, (start_x, 0), (scr_w, scr_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

        y_pos = 45
        for line in help_lines:
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
        
        cv2.putText(
            canvas, timer_text, (start_x + 15, start_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA
        )
        return canvas

