# -*- coding: utf-8 -*-
"""Модуль графического рендеринга интерфейса, OSD и окна помощи."""

import cv2
import numpy as np


class VisualRenderer(object):
    """Класс, отвечающий за компоновку финального кадра, масштабирование,

    отрисовку OSD и справочных окон Help поверх изображения.
    """

    def __init__(self, state_obj, anaglyph_processor):
        self.state = state_obj
        self.ap = anaglyph_processor

    def get_fit_to_screen_scale(self, img_w, img_h):
        """Расчет коэффициента для вписывания картинки в экран."""
        return min(self.state.screen_w / img_w, self.state.screen_h / img_h)

    def render_canvas(self, processed_img, current_filename, total_count, file_index):
        """Формирует финальный холст с OSD и справочным окном Help."""
        img_h, img_w, _ = processed_img.shape
        scr_w, scr_h = self.state.screen_w, self.state.screen_h

        new_w = max(int(img_w * self.state.scale), 1)
        new_h = max(int(img_h * self.state.scale), 1)
        resized = cv2.resize(processed_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

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

        canvas = np.zeros((scr_h, scr_w, 3), dtype=np.uint8)
        start_x = (scr_w - new_w) // 2 + self.state.offset_x
        start_y = (scr_h - new_h) // 2 + self.state.offset_y

        c_x1, c_y1 = max(0, start_x), max(0, start_y)
        c_x2, c_y2 = min(scr_w, start_x + new_w), min(scr_h, start_y + new_h)
        r_x1, r_y1 = max(0, -start_x), max(0, -start_y)
        r_x2, r_y2 = r_x1 + (c_x2 - c_x1), r_y1 + (c_y2 - c_y1)

        if (c_x2 > c_x1) and (c_y2 > c_y1):
            canvas[c_y1:c_y2, c_x1:c_x2] = resized[r_y1:r_y2, r_x1:r_x2]

        # === РЕНДЕРИНГ OSD СТОЛБЦА (TAB) ===
        if self.state.show_osd:
            canvas = self._draw_osd(canvas, current_filename, total_count, file_index)

        # === РЕНДЕРИНГ ОКНА СПРАВКИ (F1) ===
        if self.state.show_help:
            canvas = self._draw_help(canvas)

        return canvas

    def _draw_osd(self, canvas, current_filename, total_count, file_index):
        """Внутренний метод отрисовки OSD."""
        m_name = self.ap.method_details.get(self.state.method_name, self.state.method_name)
        mode_status = self.state.view_mode.upper() if self.state.has_depth else "2D FLAT IMAGE (NO 3D METADATA)"
        order_label = "SEQ" if self.state.slideshow_order == "sequential" else "RND"
        ss_status = f"ACTIVE ({order_label} | {self.state.slideshow_delay:.1f}s)" if self.state.slideshow_active else "OFF"

        osd_lines = [
            f"FILE: {current_filename} [{file_index + 1}/{total_count}]",
            f"VIEW MODE: {mode_status}",
            f"ANAGLYPH METHOD: {m_name if (self.state.view_mode == 'anaglyph' and self.state.has_depth) else 'N/A'}",
            f"GEOMETRY INTERPRETATION: {self.state.depth_interpretation.upper() if self.state.has_depth else 'N/A'}",
            f"MICRORELIEF FILTER: {'ON' if (self.state.use_microrelief and self.state.has_depth) else 'OFF'}",
            f"DISPARITY (STEREO BASE): {self.state.disparity if self.state.has_depth else '0'}",
            f"CONVERGENCE (PLANE SHIFT): {self.state.convergence}",
            f"BLUR RADIUS: {self.state.blur if self.state.has_depth else 'N/A'}",
            f"OPACITY STRENGTH: {f'{self.state.opacity:.2f}' if self.state.has_depth else 'N/A'}",
            f"SCALE: {self.state.scale * 100:.1f}% | OFFSET: {self.state.offset_x}, {self.state.offset_y}",
            f"SLIDESHOW PLAYBACK: {ss_status}",
            "------------------------------------------------",
            "Press [ TAB ] to hide OSD Panel | [ F1 ] for Interactive Help Window"
        ]
        
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 15), (580, len(osd_lines) * 26 + 25), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        y_pos = 40
        for line in osd_lines:
            cv2.putText(canvas, line, (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
            y_pos += 26
        return canvas

    def _draw_help(self, canvas):
        """Внутренний метод отрисовки всплывающего окна F1."""
        scr_w, scr_h = self.state.screen_w, self.state.screen_h
        help_lines = [
            "=== DYNAMIC DEPTH INTERACTIVE VIEWER HELP ===",
            "[ ESC ]             Exit program viewport session",
            "[ ENTER ]           Export current configured frame frame to high-quality PNG",
            "[ SPACE ]           Switch window mode to Original Flat Photo",
            "[ N ]               Switch window mode to Horizontal Side-by-Side Stereopair",
            "[ C ]               Switch window mode to Jet-Colored Depth Map Visualization",
            "--- RED-CYAN ANAGLYPH MATRICES RAMP ---",
            "[ Q W E R T Y U I O P ] -> Trigger 10 basic empirical and classic methods",
            "--- ADVANCED & GREEN-MAGENTA MATRICES RAMP ---",
            "[ A S K L ]         -> Trigger Scientific Methods (Dubois, Olson, McAllister, Rossman)",
            "[ D ]               -> Trigger LCD-Display Spectrum Optimized Thor Olson Method",
            "[ Z ]               -> Trigger Green-Magenta Empirical Pure Matrix",
            "[ X ]               -> Trigger Green-Magenta Dubois Linear Crosstalk-Compensated Method",
            "--- 3D MODIFIERS & SLIDESHOW CONTROLS ---",
            "[ M ]               -> Toggle ВЧ-Microrelief (Overlay Blend Formula) ON/OFF",
            "[ H ]               -> Toggle Geometry Interpretation (LINEAR / LOGARITHMIC)",
            "[ / ] / [ * ]       -> Scale Stereo Base down / up (Disparity step 2)",
            "[ , ] / [ . ]       -> Shift Zero Parallax Plane (Depth Convergence)",
            "[ 7 ] / [ 8 ]       -> Scale Detail Filter down / up (Blur step 2)",
            "[ 4 ] / [ 5 ]       -> Scale Detail Strength down / up (Opacity step 0.05)",
            "[ + ] / [ - ]       -> Interactive Canvas Zoom In / Zoom Out step 15%",
            "[ 0 ] / [ 1 ]       -> Reset to 100% texture scale / Auto-Fit to Screen viewport",
            "[ V ]               -> Toggle Slideshow Autoplay engine ON / OFF",
            "[ B ]               -> Toggle Slideshow Order (SEQUENTIAL / RANDOM SHUFFLE)",
            "[ [ ] / [ ] ]       -> Decrease / Increase Slideshow Autoplay timer interval by 0.5s",
            "[ ARROW KEYS ]      -> Pan/Scroll oversized matrix across viewport canvas",
            "[ J ]               -> Export complete Pydantic application session metadata dump",
        ]
        
        box_w, box_h = 790, len(help_lines) * 25 + 30
        start_x, start_y = (scr_w - box_w) // 2, (scr_h - box_h) // 2

        overlay = canvas.copy()
        cv2.rectangle(overlay, (start_x, start_y), (start_x + box_w, start_y + box_h), (15, 15, 15), -1)
        cv2.rectangle(overlay, (start_x, start_y), (start_x + box_w, start_y + box_h), (0, 0, 200), 2)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)

        y_pos = start_y + 30
        for line in help_lines:
            color = (0, 255, 255) if (line.startswith("===") or line.startswith("---")) else (255, 255, 255)
            cv2.putText(canvas, line, (start_x + 30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
            y_pos += 25
        return canvas
