# -*- coding: utf-8 -*-
"""Модуль перехвата ввода событий SDL2 для Linux: логика переключения файлов."""

import os
import random
import sys
import time
import cv2
import pygame

from anaglyph_processor import AnaglyphProcessor
from gdepth_processor import GDepthProcessor


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

        # Буфер для хранения перемешанной копии путей файлов
        self._shuffle_pool = []

    def get_current_file(self):
        return self.files[self.file_index]

    def update_processors_and_render(self):
        return self.ap.render_view_mode(self.state)

    def _generate_shuffle_pool(self):
        """Создает и перемешивает копию списка файлов, исключая текущий на старте."""
        self._shuffle_pool = list(self.files)
        random.shuffle(self._shuffle_pool)
        
        # Чтобы избежать мгновенного повтора текущей картинки при активации режима:
        if len(self._shuffle_pool) > 1 and self._shuffle_pool[-1] == self.files[self.file_index]:
            # Перемещаем её с конца в начало пула
            self._shuffle_pool.insert(0, self._shuffle_pool.pop())

    def _switch_file(self, direction):
        if len(self.files) <= 1:
            return

        # === ЛОГИКА СЛУЧАЙНОГО ВЫБОРА БЕЗ ПОВТОРЕНИЙ (ЧЕРЕЗ КЛАССИЧЕСКИЙ POP) ===
        if self.state.slideshow_active and self.state.slideshow_order == "random":
            if not self._shuffle_pool:
                self._generate_shuffle_pool()

            # Выталкиваем имя последнего файла из перемешанного массива
            next_file = self._shuffle_pool.pop()
            
            # Находим его физическое положение в основном списке, чтобы обновить OSD и счетчики
            self.file_index = self.files.index(next_file)
        else:
            # Обычный циклический режим
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
    # === ПРОДОЛЖЕНИЕ КЛАССА KeyboardController: ОБРАБОТКА ВВОДА ===

    def handle_input(self, image_cached):
        """Парсинг нативной очереди событий SDL."""
        
        # Обсчет таймера слайд-шоу
        if self.state.slideshow_active:
            current_elapsed = time.time() - self.slideshow_start_time
            self.state.slideshow_accumulator = current_elapsed
            if current_elapsed >= self.state.slideshow_delay:
                self._switch_file(1)
        else:
            self.state.slideshow_accumulator = 0.0

        # Опрос мыши для Facebook 3D Look (работает непрерывно без задержек окон)
        if self.state.fb_3d_mode:
            mx, my = pygame.mouse.get_pos()
            center_x = self.state.screen_w / 2.0
            center_y = self.state.screen_h / 2.0
            self.state.mouse_dx = -((mx - center_x) / center_x)
            self.state.mouse_dy = -((my - center_y) / center_y)
            self.state.need_update = True

        # Читаем стек прерываний SDL
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.KEYDOWN:
                key = event.key
                
                # Системные клавиши выхода и интерфейса
                if key == pygame.K_ESCAPE:
                    return False
                elif key == pygame.K_TAB:
                    self.state.show_osd = not self.state.show_osd
                elif key == pygame.K_SPACE:
                    self.state.slideshow_active = not self.state.slideshow_active
                    self.slideshow_start_time = time.time()
                    self.state.slideshow_accumulator = 0.0
                    self.state.need_update = True

                # Функциональный ряд F1-F11
                elif key == pygame.K_F1:  self.state.show_help = not self.state.show_help
                elif key == pygame.K_F3:
                    self.state.view_mode = "primary"
                    self.state.need_update = True
                elif key == pygame.K_F4:
                    self.state.view_mode = "stereopair"
                    self.state.need_update = True
                elif key == pygame.K_F5:
                    self.state.view_mode = "depthmap"
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
                    self.state.slideshow_accumulator = 0.0
                elif key == pygame.K_F10:
                    self.state.slideshow_order = "random" if self.state.slideshow_order == "sequential" else "sequential"
                    # Принудительно очищаем старый пул, чтобы при переключении собралась новая случайная колода
                    self._shuffle_pool = []
                elif key == pygame.K_F11:
                    self.state.fb_3d_mode = not self.state.fb_3d_mode
                    if not self.state.fb_3d_mode:
                        self.state.mouse_dx, self.state.mouse_dy = 0.0, 0.0
                    self.state.need_update = True

                # Экспорт кадра (F2)
                elif key == pygame.K_F2:
                    base_name = self.files[self.file_index].rsplit('.', 1)[0]
                    m_suffix = self.state.view_mode.upper() if self.state.view_mode != "anaglyph" else self.state.method_name
                    out_filename = f"{base_name}_{m_suffix}_sdl.png"
                    try:
                        cv2.imwrite(out_filename, image_cached)
                        print(f"[+] Экспорт завершен: {out_filename}")
                    except Exception as e: print(f"[-] Ошибка экспорта: {e}")

                # Навигация файлов (PageUp / PageDown)
                elif key == pygame.K_PAGEUP:   self._switch_file(-1)
                elif key == pygame.K_PAGEDOWN: self._switch_file(1)

                # Цветовые матрицы анаглифов (Строго по SDL маппингу литер)
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

                # Настройка слайд-шоу задержки
                elif key == pygame.K_LEFTBRACKET:
                    self.state.slideshow_delay = max(0.5, self.state.slideshow_delay - 0.5)
                elif key == pygame.K_RIGHTBRACKET:
                    self.state.slideshow_delay = min(30.0, self.state.slideshow_delay + 0.5)

                # Управление параметрами 3D через стандартную цифровую строку клавиатуры
                elif key == pygame.K_KP_MULTIPLY or key == pygame.K_8:
                    self.state.disparity = min(100, self.state.disparity + 2)
                    self.state.need_update = True
                elif key == pygame.K_KP_DIVIDE or key == pygame.K_7:
                    self.state.disparity = max(0, self.state.disparity - 2)
                    self.state.need_update = True
                elif key == pygame.K_INSERT or key == pygame.K_6:
                    self.state.convergence = min(50, self.state.convergence + 2)
                    self.state.need_update = True
                elif key == pygame.K_DELETE or key == pygame.K_5:
                    self.state.convergence = max(-50, self.state.convergence - 2)
                    self.state.need_update = True

                # Масштабирование холста
                elif key in (pygame.K_PLUS, pygame.K_EQUALS):  self.state.scale *= 1.05
                elif key == pygame.K_MINUS:                     self.state.scale /= 1.05
                elif key == pygame.K_0:
                    self.state.scale = 1.0
                    self.state.offset_x, self.state.offset_y = 0, 0
                elif key == pygame.K_1:
                    h, w, _ = image_cached.shape
                    self.state.scale = self.renderer.get_fit_to_screen_scale(w, h)
                    self.state.offset_x, self.state.offset_y = 0, 0

                # Физические стрелки панорамирования матрицы
                elif key == pygame.K_LEFT:  self.state.offset_x += self.move_step
                elif key == pygame.K_RIGHT: self.state.offset_x -= self.move_step
                elif key == pygame.K_UP:    self.state.offset_y += self.move_step
                elif key == pygame.K_DOWN:  self.state.offset_y -= self.move_step

        return True

    def _set_anaglyph(self, method):
        self.state.view_mode = "anaglyph"
        self.state.method_name = method
        self.state.need_update = True

