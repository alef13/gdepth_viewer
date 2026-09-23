# -*- coding: utf-8 -*-
"""Модуль асинхронного перехвата ввода и управления сессией слайд-шоу."""

import ctypes
import os
import random
import sys
import time
import cv2

from anaglyph_processor import AnaglyphProcessor
from gdepth_processor import GDepthProcessor


class KeyboardController(object):
    """Класс-контроллер для обработки ввода, таймеров слайд-шоу и переключения файлов.

    Абсолютно все горячие клавиши переведены на физические коды WinAPI Virtual-Key,
    что делает управление независимым от раскладки и защищает от синтаксических ошибок.
    """

    def __init__(self, files_list, start_idx, window_name, anaglyph_processor, state_obj, visual_renderer):
        self.files = files_list
        self.file_index = start_idx
        self.win_name = window_name
        self.ap = anaglyph_processor
        self.state = state_obj
        self.renderer = visual_renderer
        
        # --- ПОЛНАЯ ТАБЛИЦА СИСТЕМНЫХ АППАРАТНЫХ КОНСТАНТ (VIRTUAL-KEY CODES) ---
        self.VK_LEFT, self.VK_UP, self.VK_RIGHT, self.VK_DOWN = 0x25, 0x26, 0x27, 0x28
        self.VK_PRIOR, self.VK_NEXT = 0x21, 0x22  # PgUp / PgDown
        self.VK_TAB = 0x09
        self.VK_INSERT, self.VK_DELETE = 0x2D, 0x2E
        
        # Функциональный ряд F1-F11
        self.VK_F1, self.VK_F2, self.VK_F3, self.VK_F4, self.VK_F5 = 0x70, 0x71, 0x72, 0x73, 0x74
        self.VK_F6, self.VK_F7, self.VK_F8, self.VK_F9, self.VK_F10 = 0x75, 0x76, 0x77, 0x78, 0x79
        self.VK_F11 = 0x7A  
        
        # Цифровой блок масштаба и параметров (0, 1, 4, 5, 7, 8)
        self.VK_0, self.VK_1 = 0x30, 0x31
        self.VK_4, self.VK_5 = 0x34, 0x35
        self.VK_7, self.VK_8 = 0x37, 0x38
        
        # Знаки, скобки и калькуляторный блок
        self.VK_OEM_PLUS, self.VK_OEM_MINUS = 0xBB, 0xBD  # Кнопки + и -
        self.VK_MULTIPLY, self.VK_DIVIDE = 0x6A, 0x6F    # NumPad * и /
        self.VK_OEM_2 = 0xBF                             # Физическая клавиша '/' (возле Ю)
        self.VK_OEM_4 = 0xDB                             # Физическая клавиша '[' (хетч Х)
        self.VK_OEM_6 = 0xDD                             # Физическая клавиша ']' (хетч Ъ)
        
        # Словарь key_states инициализируется гарантированно до вызова любых циклов
        self.key_states = {
            self.VK_TAB: False, self.VK_PRIOR: False, self.VK_NEXT: False,
            self.VK_F1: False, self.VK_F2: False, self.VK_F3: False, self.VK_F4: False, self.VK_F5: False,
            self.VK_F6: False, self.VK_F7: False, self.VK_F8: False, self.VK_F9: False, self.VK_F10: False,
            self.VK_F11: False,  
            self.VK_OEM_4: False, self.VK_OEM_6: False
        }
        
        # Физическая карта буквенных клавиш (ASCII-коды заглавных букв)
        self.vk_method_map = {
            0x51: "wimmer_pure",               # Q
            0x57: "wimmer_balanced",           # W
            0x45: "vu_tran_least_squares",     # E
            0x52: "wimmer_optimized_matrix",   # R
            0x54: "ducos_classic",             # T
            0x59: "sgi_gray",                  # Y
            0x55: "wimmer_half_color",         # U
            0x49: "wimmer_green_optimized",    # I
            0x4F: "dubois_projection",         # O
            0x50: "thor_olson_optimized",      # P
            0x41: "mcallister_midpoint",       # A
            0x53: "rossman_luminance",         # S
            0x44: "thor_olson_lcd",            # D
            0x5A: "green_magenta_pure",        # Z
            0x58: "green_magenta_dubois"       # X
        }
        
        # Заполняем состояния для буквенной карты
        for vk_code in self.vk_method_map:
            self.key_states[vk_code] = False

        self.get_async_key = ctypes.windll.user32.GetAsyncKeyState
        self.get_async_key.argtypes = [ctypes.c_int]
        self.get_async_key.restype = ctypes.c_short
        
        self.move_step = 25
        self.slideshow_start_time = time.time()
        self.state.slideshow_accumulator = 0.0

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

        self.slideshow_start_time = time.time()
        self.state.slideshow_accumulator = 0.0
        try:
            dp = GDepthProcessor(self.files[self.file_index])
            self.ap = AnaglyphProcessor(dp)
            self.state.has_depth = dp.has_depth
            img_h, img_w, _ = dp.primary_img.shape
            
            self.state.scale = self.renderer.get_fit_to_screen_scale(img_w, img_h)
            self.state.offset_x, self.state.offset_y = 0, 0
            self.state.need_update = True
        except Exception as err:
            print(f"[-] Ошибка загрузки {self.files[self.file_index]}: {err}", file=sys.stderr)

    def _is_pressed(self, vk_code):
        """Проверяет нажатие клавиши с защитой от повторного срабатывания (дребезга)."""
        if self.get_async_key(vk_code) < 0:
            if not self.key_states.get(vk_code, False):
                self.key_states[vk_code] = True
                return True
        else:
            self.key_states[vk_code] = False
        return False
    def mouse_callback(self, event, x, y, flags, param):
        """Отслеживает перемещение мыши для расчета динамического параллакса Facebook 3D."""
        if self.state.fb_3d_mode and event == cv2.EVENT_MOUSEMOVE:
            scr_w, scr_h = self.state.screen_w, self.state.screen_h
            center_x = scr_w / 2.0
            center_y = scr_h / 2.0
            
            self.state.mouse_dx = -((x - center_x) / center_x)
            self.state.mouse_dy = -((y - center_y) / center_y)
            self.state.need_update = True

    def handle_input(self, image_cached):
        """Главный диспетчер обработки сигналов ввода и таймера слайд-шоу."""
        loop_delay_ms = 15
        raw_key = cv2.waitKey(loop_delay_ms)
        
        if self.state.slideshow_active:
            current_elapsed = time.time() - self.slideshow_start_time
            self.state.slideshow_accumulator = current_elapsed
            if current_elapsed >= self.state.slideshow_delay:
                self._switch_file(1)
        else:
            self.state.slideshow_accumulator = 0.0

        if self.get_async_key(0x1B) < 0:  # ESC
            return False

        # --- СКАНИРОВАНИЕ МАТРИЦ АНАГЛИФА ПО ХАРДВЕРНЫМ КОДАМ БУКВЕННЫХ КЛАВИШ ---
        for vk_code, method_name in self.vk_method_map.items():
            if self._is_pressed(vk_code):
                self.state.view_mode = "anaglyph"
                self.state.method_name = method_name
                self.state.need_update = True

        # --- СКВОЗНОЙ КНОПОЧНЫЙ ОПРОС РЯДА F1-F11 И МОДИФИКАТОРОВ ---
        if self._is_pressed(self.VK_F1):
            self.state.show_help = not self.state.show_help

        if self._is_pressed(self.VK_F2):
            base_name = self.files[self.file_index].rsplit('.', 1)
            m_suffix = "Original" if self.state.view_mode == "primary" else ("Stereopair" if self.state.view_mode == "stereopair" else f"DepthMap_{self.state.depth_interpretation.upper()}" if self.state.view_mode == "depthmap" else self.ap.method_details[self.state.method_name].replace(" ", "_"))
            out_filename = f"{base_name}_{m_suffix}_d{self.state.disparity}_b{self.state.blur}_o{self.state.opacity:.2f}.png"
            try:
                cv2.imwrite(out_filename, image_cached)
                print(f"[+] Экспорт завершен: {out_filename}")
            except Exception as e:
                print(f"[-] Ошибка экспорта: {e}", file=sys.stderr)

        if self._is_pressed(self.VK_F3):
            if self.state.view_mode != "primary":
                self.state.view_mode = "primary"
                self.state.need_update = True

        if self._is_pressed(self.VK_F4):
            if self.state.view_mode != "stereopair":
                self.state.view_mode = "stereopair"
                self.state.need_update = True

        if self._is_pressed(self.VK_F5):
            if self.state.view_mode != "depthmap":
                self.state.view_mode = "depthmap"
                self.state.need_update = True

        if self._is_pressed(self.VK_F6):
            self.state.depth_interpretation = "logarithmic" if self.state.depth_interpretation == "linear" else "linear"
            self.state.need_update = True

        if self._is_pressed(self.VK_F7):
            self.state.use_microrelief = not self.state.use_microrelief
            self.state.need_update = True

        if self._is_pressed(self.VK_F8):
            current_dump = self.state.model_dump()
            dump_info = f"\n=== PYDANTIC ДАМП СОСТОЯНИЯ ===\nФайл: {self.files[self.file_index]}\n"
            for k, v in current_dump.items(): dump_info += f"{k}: {v}\n"
            dump_info += "===============================\n"
            with open("viewer_dump.txt", "a", encoding="utf-8") as df: df.write(dump_info)
            print("[+] Параметры сессии сохранены в файл viewer_dump.txt")

        if self._is_pressed(self.VK_F9):
            self.state.slideshow_active = not self.state.slideshow_active
            self.slideshow_start_time = time.time()
            self.state.slideshow_accumulator = 0.0

        if self._is_pressed(self.VK_F10):
            self.state.slideshow_order = "random" if self.state.slideshow_order == "sequential" else "sequential"

        # F11 - Включение / выключение интерактивного параллакса Facebook 3D Look
        if self._is_pressed(self.VK_F11):
            self.state.fb_3d_mode = not self.state.fb_3d_mode
            if self.state.fb_3d_mode:
                cv2.setMouseCallback(self.win_name, self.mouse_callback)
                print("[*] Режим Facebook 3D Look: АКТИВИРОВАН")
            else:
                cv2.setMouseCallback(self.win_name, lambda *args: None)
                self.state.mouse_dx = 0.0
                self.state.mouse_dy = 0.0
                print("[*] Режим Facebook 3D Look: ВЫКЛЮЧЕН")
            self.state.need_update = True

        # Изменение интервала слайд-шоу переведено на self.state
        if self._is_pressed(self.VK_OEM_4):  # Кнопка [
            try:
                self.state.slideshow_delay = max(0.5, self.state.slideshow_delay - 0.5)
                print(f"[*] Интервал слайд-шоу уменьшен: {self.state.slideshow_delay:.1f} сек.")
            except Exception: pass

        if self._is_pressed(self.VK_OEM_6):  # Кнопка ]
            try:
                self.state.slideshow_delay = min(30.0, self.state.slideshow_delay + 0.5)
                print(f"[*] Интервал слайд-шоу увеличен: {self.state.slideshow_delay:.1f} сек.")
            except Exception: pass

        # Регулировка шага стереобазы (Disparity) по символам '*' и '/'
        if self.get_async_key(self.VK_MULTIPLY) < 0 or (self.get_async_key(0x10) < 0 and self.get_async_key(0x38) < 0):
            try: self.state.disparity = min(100, self.state.disparity + 2); self.state.need_update = True
            except Exception: pass
        elif self.get_async_key(self.VK_DIVIDE) < 0 or self.get_async_key(self.VK_OEM_2) < 0:
            try: self.state.disparity = max(0, self.state.disparity - 2); self.state.need_update = True
            except Exception: pass

        # Регулировка плоскости схождения (Convergence) на аппаратные клавиши INS / DEL
        if self.get_async_key(self.VK_INSERT) < 0:
            try: self.state.convergence = min(50, self.state.convergence + 2); self.state.need_update = True
            except Exception: pass
        elif self.get_async_key(self.VK_DELETE) < 0:
            try: self.state.convergence = max(-50, self.state.convergence - 2); self.state.need_update = True
            except Exception: pass

        # Изменение размытия (Blur 7/8) и силы микрорельефа (Opacity 4/5)
        if self.get_async_key(self.VK_8) < 0:
            try: self.state.blur = min(101, self.state.blur + 2); self.state.need_update = True
            except Exception: pass
        elif self.get_async_key(self.VK_7) < 0:
            try: self.state.blur = max(3, self.state.blur - 2); self.state.need_update = True
            except Exception: pass

        if self.get_async_key(self.VK_5) < 0:
            try: self.state.opacity = round(min(1.0, self.state.opacity + 0.05), 2); self.state.need_update = True
            except Exception: pass
        elif self.get_async_key(self.VK_4) < 0:
            try: self.state.opacity = round(max(0.0, self.state.opacity - 0.05), 2); self.state.need_update = True
            except Exception: pass

        # Интерактивное масштабирование кадра (+ / - / 0 / 1)
        if self.get_async_key(self.VK_OEM_PLUS) < 0: self.state.scale *= 1.05
        elif self.get_async_key(self.VK_OEM_MINUS) < 0: self.state.scale /= 1.05
        elif self.get_async_key(self.VK_0) < 0:
            self.state.scale = 1.0
            self.state.offset_x, self.state.offset_y = 0, 0
        elif self.get_async_key(self.VK_1) < 0:
            h, w, _ = image_cached.shape
            self.state.scale = self.renderer.get_fit_to_screen_scale(w, h)
            self.state.offset_x, self.state.offset_y = 0, 0

        # Перелистывание каталога изображений PageUp / PageDown
        if self._is_pressed(self.VK_PRIOR): self._switch_file(-1)
        if self._is_pressed(self.VK_NEXT): self._switch_file(1)
        
        # Переключение OSD по нажатию TAB
        if self._is_pressed(self.VK_TAB): self.state.show_osd = not self.state.show_osd

        # Навигация панорамирования физическими стрелками клавиатуры
        if self.get_async_key(self.VK_LEFT) < 0: self.state.offset_x += self.move_step
        if self.get_async_key(self.VK_RIGHT) < 0: self.state.offset_x -= self.move_step
        if self.get_async_key(self.VK_UP) < 0: self.state.offset_y += self.move_step
        if self.get_async_key(self.VK_DOWN) < 0: self.state.offset_y -= self.move_step

        return True
