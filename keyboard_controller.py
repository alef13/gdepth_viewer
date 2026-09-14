# -*- coding: utf-8 -*-
"""Модуль асинхронного перехвата ввода и управления сессией слайд-шоу."""

import ctypes
import os
import random
import sys
import cv2

from anaglyph_processor import AnaglyphProcessor
from gdepth_processor import GDepthProcessor


class KeyboardController(object):
    """Класс-контроллер для обработки ввода, таймеров слайд-шоу и переключения файлов."""

    def __init__(self, files_list, start_idx, window_name, anaglyph_processor, state_obj, visual_renderer):
        self.files = files_list
        self.file_index = start_idx
        self.win_name = window_name
        self.ap = anaglyph_processor
        self.state = state_obj
        self.renderer = visual_renderer  # Инжектируем класс визуализации
        
        # Кроссплатформенные константы клавиш WinAPI
        self.VK_LEFT, self.VK_UP, self.VK_RIGHT, self.VK_DOWN = 0x25, 0x26, 0x27, 0x28
        self.VK_PRIOR, self.VK_NEXT = 0x21, 0x22
        self.VK_TAB, self.VK_F1 = 0x09, 0x70
        
        self.get_async_key = ctypes.windll.user32.GetAsyncKeyState
        self.get_async_key.argtypes = [ctypes.c_int]
        self.get_async_key.restype = ctypes.c_short

        self.pgup_pressed = False
        self.pgdn_pressed = False
        self.tab_pressed = False
        self.f1_pressed = False
        self.v_pressed = False
        self.b_pressed = False
        self.h_pressed = False
        self.j_pressed = False
        self.move_step = 25
        self.slideshow_accumulator = 0

        self.key_method_map = {
            "q": "wimmer_pure", "w": "wimmer_balanced", "e": "vu_tran_least_squares",
            "r": "wimmer_optimized_matrix", "t": "ducos_classic", "y": "sgi_gray",
            "u": "wimmer_half_color", "i": "wimmer_green_optimized", "o": "dubois_projection",
            "p": "thor_olson_optimized",
            "a": "mcallister_midpoint", "s": "rossman_luminance", "d": "thor_olson_lcd",
            "z": "green_magenta_pure", "x": "green_magenta_dubois"
        }

    def get_current_file(self):
        return self.files[self.file_index]

    def update_processors_and_render(self):
        return self.ap.render_view_mode(self.state)

    def _switch_file(self, direction):
        if self.state.slideshow_active and self.state.slideshow_order == "random" and len(self.files) > 1:
            next_idx = self.file_index
            while next_idx == self.file_index:
                next_idx = random.randint(0, len(self.files) - 1)
            self.file_index = next_idx
        else:
            self.file_index = (self.file_index + direction) % len(self.files)

        self.slideshow_accumulator = 0
        try:
            dp = GDepthProcessor(self.files[self.file_index])
            self.ap = AnaglyphProcessor(dp)
            self.state.has_depth = dp.has_depth
            img_h, img_w, _ = dp.primary_img.shape
            
            # Вызов утилиты вычисления масштаба из VisualRenderer
            self.state.scale = self.renderer.get_fit_to_screen_scale(img_w, img_h)
            self.state.offset_x, self.state.offset_y = 0, 0
            self.state.need_update = True
        except Exception as err:
            print(f"[-] Ошибка загрузки {self.files[self.file_index]}: {err}", file=sys.stderr)

    def handle_input(self, image_cached):
        """Главный диспетчер обработки сигналов ввода и таймера слайд-шоу."""
        loop_delay_ms = 15
        raw_key = cv2.waitKey(loop_delay_ms)
        
        if self.state.slideshow_active:
            self.slideshow_accumulator += loop_delay_ms
            if self.slideshow_accumulator >= (self.state.slideshow_delay * 1000):
                self._switch_file(1)

        if raw_key == 27:  # ESC
            return False

        if raw_key != -1:
            key_8bit = raw_key & 0xFF
            char_key = chr(key_8bit).lower() if 32 <= key_8bit < 127 else ""

            if key_8bit == 32:  # Пробел
                if self.state.view_mode != "primary":
                    self.state.view_mode = "primary"; self.state.need_update = True
            elif char_key == "n":  # Режим горизонтальной стереопары Side-by-Side
                if self.state.view_mode != "stereopair":
                    self.state.view_mode = "stereopair"; self.state.need_update = True
            elif char_key == "c":  # Режим визуализации карты глубин
                if self.state.view_mode != "depthmap":
                    self.state.view_mode = "depthmap"; self.state.need_update = True
            elif char_key == "m":  # Включение/отключение микрорельефа
                self.state.use_microrelief = not self.state.use_microrelief; self.state.need_update = True
            elif char_key in self.key_method_map:
                self.state.view_mode = "anaglyph"
                self.state.method_name = self.key_method_map[char_key]
                self.state.need_update = True

            elif key_8bit in (ord("+"), ord("=")): self.state.scale *= 1.15
            elif key_8bit == ord("-"): self.state.scale /= 1.15
            elif key_8bit == ord("0"):
                self.state.scale = 1.0; self.state.offset_x, self.state.offset_y = 0, 0
            elif key_8bit == ord("1"):
                h, w, _ = image_cached.shape
                self.state.scale = self.renderer.get_fit_to_screen_scale(w, h)
                self.state.offset_x, self.state.offset_y = 0, 0

            elif char_key == "[":
                try: self.state.slideshow_delay = max(0.5, self.state.slideshow_delay - 0.5)
                except Exception: pass
            elif char_key == "]":
                try: self.state.slideshow_delay = min(30.0, self.state.slideshow_delay + 0.5)
                except Exception: pass

            elif key_8bit == ord("/"):
                try: self.state.disparity = max(0, self.state.disparity - 2); self.state.need_update = True
                except Exception: pass
            elif key_8bit == ord("*"):
                try: self.state.disparity = min(100, self.state.disparity + 2); self.state.need_update = True
                except Exception: pass
            elif key_8bit == ord("7"):
                try: self.state.blur = max(3, self.state.blur - 2); self.state.need_update = True
                except Exception: pass
            elif key_8bit == ord("8"):
                try: self.state.blur = min(101, self.state.blur + 2); self.state.need_update = True
                except Exception: pass
            elif key_8bit == ord("4"):
                try: self.state.opacity = round(max(0.0, self.state.opacity - 0.05), 2); self.state.need_update = True
                except Exception: pass
            elif key_8bit == ord("5"):
                try: self.state.opacity = round(min(1.0, self.state.opacity + 0.05), 2); self.state.need_update = True
                except Exception: pass
            # Регулировка положительного/отрицательного параллакса (Плоскость конвергенции)
            elif char_key == ",":
                try:
                    self.state.convergence = max(-50, self.state.convergence - 5)
                    self.state.need_update = True
                    print(f"[*] Схождение сдвинуто назад (больше положительного параллакса): {self.state.convergence}")
                except Exception: pass
            elif char_key == ".":
                try:
                    self.state.convergence = min(50, self.state.convergence + 5)
                    self.state.need_update = True
                    print(f"[*] Схождение сдвинуто вперед (больше вылета из экрана): {self.state.convergence}")
                except Exception: pass


            elif key_8bit == 13:  # Enter
                base_name = self.files[self.file_index].rsplit('.', 1)[0]
                m_suffix = "Original" if self.state.view_mode == "primary" else ("Stereopair" if self.state.view_mode == "stereopair" else f"DepthMap_{self.state.depth_interpretation.upper()}" if self.state.view_mode == "depthmap" else self.ap.method_details[self.state.method_name].replace(" ", "_"))
                out_filename = f"{base_name}_{m_suffix}_d{self.state.disparity}_b{self.state.blur}_o{self.state.opacity:.2f}.png"
                try: cv2.imwrite(out_filename, image_cached)
                except Exception: pass

        # Низкоуровневый асинхронный опрос WinAPI
        if self.get_async_key(ord("V")) < 0:
            if not self.v_pressed: self.state.slideshow_active = not self.state.slideshow_active; self.slideshow_accumulator = 0; self.v_pressed = True
        else: self.v_pressed = False

        if self.get_async_key(ord("B")) < 0:
            if not self.b_pressed: self.state.slideshow_order = "random" if self.state.slideshow_order == "sequential" else "sequential"; self.b_pressed = True
        else: self.b_pressed = False

        if self.get_async_key(ord("H")) < 0:
            if not self.h_pressed: self.state.depth_interpretation = "logarithmic" if self.state.depth_interpretation == "linear" else "linear"; self.state.need_update = True; self.h_pressed = True
        else: self.h_pressed = False

        if self.get_async_key(ord("J")) < 0:
            if not self.j_pressed:
                current_dump = self.state.model_dump()
                dump_info = f"\n=== PYDANTIC ДАМП СОСТОЯНИЯ ===\nФайл: {self.files[self.file_index]}\n"
                for k, v in current_dump.items(): dump_info += f"{k}: {v}\n"
                dump_info += "===============================\n"
                with open("viewer_dump.txt", "a", encoding="utf-8") as df: df.write(dump_info)
                self.j_pressed = True
        else: self.j_pressed = False

        if self.get_async_key(self.VK_TAB) < 0:
            if not self.tab_pressed: self.state.show_osd = not self.state.show_osd; self.tab_pressed = True
        else: self.tab_pressed = False

        if self.get_async_key(self.VK_F1) < 0:
            if not self.f1_pressed: self.state.show_help = not self.state.show_help; self.f1_pressed = True
        else: self.f1_pressed = False

        if self.get_async_key(self.VK_PRIOR) < 0:
            if not self.pgup_pressed: self._switch_file(-1); self.pgup_pressed = True
        else: self.pgup_pressed = False

        if self.get_async_key(self.VK_NEXT) < 0:
            if not self.pgdn_pressed: self._switch_file(1); self.pgdn_pressed = True
        else: self.pgdn_pressed = False

        if self.get_async_key(self.VK_LEFT) < 0: self.state.offset_x += self.move_step
        if self.get_async_key(self.VK_RIGHT) < 0: self.state.offset_x -= self.move_step
        if self.get_async_key(self.VK_UP) < 0: self.state.offset_y += self.move_step
        if self.get_async_key(self.VK_DOWN) < 0: self.state.offset_y -= self.move_step

        return True
