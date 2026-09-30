# -*- coding: utf-8 -*-
"""Модуль перехвата ввода событий SDL2 для Linux. Часть 1: Смена файлов."""

import os
import random
import sys
import time
import cv2
import pygame
import pygame_gui

from anaglyph_processor import AnaglyphProcessor
from gdepth_processor import GDepthProcessor
from lenticular_generator import LenticularGenerator
from lenticular_ui import LenticularUI


class KeyboardController(object):
    """Класс-контроллер для обработки событий мыши и клавиатуры через SDL-шину."""

    def __init__(self, files_list, start_idx, anaglyph_processor, state_obj, visual_renderer):
        self.files = files_list
        self.file_index = start_idx
        self.ap = anaglyph_processor
        self.state = state_obj
        self.renderer = visual_renderer
        
        self.move_step = 25
        self.slideshow_start_time = time.time()
        self.state.slideshow_accumulator = 0.0
        self._shuffle_pool = []

        # Инжектируем изолированный GUI компонент
        self.ui = LenticularUI(self.state.screen_w, self.state.screen_h)
        self.generation_thread = None

    def get_current_file(self):
        return self.files[self.file_index]

    def update_processors_and_render(self):
        return self.ap.render_view_mode(self.state)

    def _generate_shuffle_pool(self):
        self._shuffle_pool = list(self.files)
        random.shuffle(self._shuffle_pool)
        if len(self._shuffle_pool) > 1 and self._shuffle_pool[-1] == self.files[self.file_index]:
            self._shuffle_pool.insert(0, self._shuffle_pool.pop())

    def _switch_file(self, direction):
        if len(self.files) <= 1: return
        if self.state.slideshow_active and self.state.slideshow_order == "random":
            if not self._shuffle_pool: self._generate_shuffle_pool()
            self.file_index = self.files.index(self._shuffle_pool.pop())
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
    # === ПРОДОЛЖЕНИЕ КЛАССА KeyboardController: ЦИКЛ ОБРАБОТКИ СОБЫТИЙ И ХОТКЕИ ===

    def handle_input(self, image_cached):
        """Парсинг единой нативной очереди событий SDL."""
        
        # Обсчет таймера слайд-шоу (активен, если не открыто окно настройки)
        if self.state.slideshow_active and not self.ui.window and not self.ui.is_generating:
            current_elapsed = time.time() - self.slideshow_start_time
            self.state.slideshow_accumulator = current_elapsed
            if current_elapsed >= self.state.slideshow_delay: 
                self._switch_file(1)
                self.state.need_update = True
        elif not self.state.slideshow_active:
            self.state.slideshow_accumulator = 0.0

        # Мониторинг координат для интерактива Facebook 3D Look
        if self.state.fb_3d_mode and not self.ui.window and not self.ui.is_generating:
            mx, my = pygame.mouse.get_pos()
            self.state.mouse_dx = -((mx - self.state.screen_w/2.0) / (self.state.screen_w/2.0))
            self.state.mouse_dy = -((my - self.state.screen_h/2.0) / (self.state.screen_h/2.0))
            self.state.need_update = True

        time_delta = pygame.time.Clock().tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            
            # Всегда транслируем прерывания в стек pygame_gui менеджера
            self.ui.manager.process_events(event)
                
            if self.ui.window:
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.ui.btn_cancel: self.ui.close_dialog()
                    elif event.ui_element == self.ui.btn_start: self._dispatch_generation()
                continue

            if event.type == pygame.KEYDOWN and not self.ui.is_generating:
                key = event.key
                if key == pygame.K_ESCAPE: return False
                elif key == pygame.K_TAB:   self.state.show_osd = not self.state.show_osd
                elif key == pygame.K_SPACE:
                    self.state.slideshow_active = not self.state.slideshow_active
                    self.slideshow_start_time = time.time()
                    self.state.need_update = True
                elif key == pygame.K_F1:  self.state.show_help = not self.state.show_help
                elif key in (pygame.K_F3, pygame.K_F4, pygame.K_F5):
                    self.state.view_mode = "primary" if key == pygame.K_F3 else "stereopair" if key == pygame.K_F4 else "depthmap"
                    self.state.need_update = True
                elif key == pygame.K_F6:
                    self.state.depth_interpretation = "logarithmic" if self.state.depth_interpretation == "linear" else "linear"
                    self.state.need_update = True
                elif key == pygame.K_F7:
                    self.state.use_microrelief = not self.state.use_microrelief
                    self.state.need_update = True
                elif key == pygame.K_F9:
                    self.state.slideshow_active = not self.state.slideshow_active
                    self.slideshow_start_time = time.time()
                    self.state.need_update = True
                elif key == pygame.K_F10:
                    self.state.slideshow_order = "random" if self.state.slideshow_order == "sequential" else "sequential"
                    self._shuffle_pool = []
                elif key == pygame.K_F11:
                    self.state.fb_3d_mode = not self.state.fb_3d_mode
                    if not self.state.fb_3d_mode: self.state.mouse_dx, self.state.mouse_dy = 0.0, 0.0
                    self.state.need_update = True
                elif key == pygame.K_F12:
                    self.state.slideshow_active = False
                    self.ui.show_dialog()
                elif key == pygame.K_PAGEUP:   self._switch_file(-1)
                elif key == pygame.K_PAGEDOWN: self._switch_file(1)
                
                # Матрицы микширования 3D-каналов анаглифов
                elif key == pygame.K_q: self._set_anaglyph("wimmer_pure")
                elif key == pygame.K_w: self._set_anaglyph("wimmer_balanced")
                elif key == pygame.K_e: self._set_anaglyph("vu_tran_least_squares")
                elif key == pygame.K_r: self._set_anaglyph("wimmer_optimized_matrix")
                elif key == pygame.K_t: self._set_anaglyph("ducos_classic")
                elif key == pygame.K_y: self._set_anaglyph("sgi_gray")
                elif key == pygame.K_u: self._set_anaglyph("wimmer_half_color")
                elif key == pygame.K_i: self._set_anaglyph("wimmer_green_optimized")
                elif key == pygame.K_o: self._set_anaglyph("dubois_projection")
                elif key == pygame.K_p: self._set_anaglyph("thor_olson_optimized")
                elif key == pygame.K_a: self._set_anaglyph("mcallister_midpoint")
                elif key == pygame.K_s: self._set_anaglyph("rossman_luminance")
                elif key == pygame.K_d: self._set_anaglyph("thor_olson_lcd")
                elif key == pygame.K_z: self._set_anaglyph("green_magenta_pure")
                elif key == pygame.K_x: self._set_anaglyph("green_magenta_dubois")
                
                # Переход к Блоку №3 для продолжения обработки клавиш масштаба...
                # === ПРОДОЛЖЕНИЕ ХОТКЕЕВ: МАСШТАБ, СТРЕЛКИ И ВОРКЕРЫ ===
                elif key == pygame.K_LEFTBRACKET:
                    self.state.slideshow_delay = max(0.5, self.state.slideshow_delay - 0.5)
                elif key == pygame.K_RIGHTBRACKET:
                    self.state.slideshow_delay = min(30.0, self.state.slideshow_delay + 0.5)

                elif key in (pygame.K_KP_MULTIPLY, pygame.K_8):
                    self.state.disparity = min(100, self.state.disparity + 2)
                    self.state.need_update = True
                elif key in (pygame.K_KP_DIVIDE, pygame.K_7):
                    self.state.disparity = max(0, self.state.disparity - 2)
                    self.state.need_update = True
                elif key == pygame.K_INSERT or key == pygame.K_6:
                    self.state.convergence = min(50, self.state.convergence + 2)
                    self.state.need_update = True
                elif key == pygame.K_DELETE or key == pygame.K_5:
                    self.state.convergence = max(-50, self.state.convergence - 2)
                    self.state.need_update = True
                
                elif key in (pygame.K_PLUS, pygame.K_EQUALS):  
                    self.state.scale *= 1.05
                    self.state.need_update = True
                elif key == pygame.K_MINUS:                     
                    self.state.scale /= 1.05
                    self.state.need_update = True
                elif key == pygame.K_0:
                    self.state.scale = 1.0
                    self.state.offset_x, self.state.offset_y = 0, 0
                    self.state.need_update = True
                elif key == pygame.K_1:
                    h, w, _ = self.ap.dp.primary_img.shape
                    self.state.scale = self.renderer.get_fit_to_screen_scale(w, h)
                    self.state.offset_x, self.state.offset_y = 0, 0
                    self.state.need_update = True

                elif key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
                    step = self.move_step
                    if key == pygame.K_LEFT: self.state.offset_x += step
                    elif key == pygame.K_RIGHT: self.state.offset_x -= step
                    elif key == pygame.K_UP: self.state.offset_y += step
                    elif key == pygame.K_DOWN: self.state.offset_y -= step
                    self.state.need_update = True

        return True

    def _dispatch_generation(self):
        """Асинхронный диспетчер запуска вычислительного ядра генератора."""
        import threading
        try:
            params = self.ui.get_parsed_params(self.ap.dp.primary_img.copy())
            self.ui.close_dialog()
            self.ui.is_generating = True
            self.ui.progress_val = 0.0
            self.ui.progress_text = "Spawning worker..."

            self.generation_thread = threading.Thread(target=self._async_worker, args=(params,))
            self.generation_thread.daemon = True
            self.generation_thread.start()
        except ValueError: print("[-] Ошибка чтения UI параметров.")

    def _async_worker(self, params):
        try:
            generator = LenticularGenerator(params, self.state, self.ap)
            result = generator.run(progress_callback=lambda v, t: setattr(self.ui, 'progress_val', v) or setattr(self.ui, 'progress_text', t))
            
            base_name = self.files[self.file_index].rsplit('.', 1)[0]
            out_fn = f"{base_name}_LENTICULAR_{int(params['lpi'])}LPI_{params['dpi']}DPI.png"
            cv2.imwrite(out_fn, result)
            print(f"[+] Успешно: {out_fn}")
        except Exception as e: print(f"[-] Ошибка: {e}")
        finally:
            self.ui.is_generating = False

    def draw_ui_and_progress(self, screen_surface):
        """Пробрасывает вызов рендеринга на изолированный UI-модуль."""
        self.ui.draw(screen_surface)
        if self.generation_thread and not self.generation_thread.is_alive():
            self.generation_thread = None
            self.state.need_update = True

    def _set_anaglyph(self, method):
        self.state.view_mode = "anaglyph"
        self.state.method_name = method
        self.state.need_update = True

