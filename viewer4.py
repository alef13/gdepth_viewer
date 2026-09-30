# -*- coding: utf-8 -*-
"""Главный интерфейсный модуль интерактивного 3D-просмотрщика на SDL2."""

import argparse
import os
import sys
import pygame
import numpy as np

from anaglyph_processor import AnaglyphProcessor
from config import ViewerState
from gdepth_processor import GDepthProcessor
from keyboard_controller import KeyboardController
from visual_renderer import VisualRenderer

state = ViewerState()

def parse_cli_args():
    parser = argparse.ArgumentParser(description="ООП-просмотрщик Dynamic Depth на SDL.")
    parser.add_argument("file", help="Путь к файлу изображения JPEG")
    parser.add_argument("--mode", "-mode", choices=["anaglyph", "stereopair", "depthmap", "primary"], default="anaglyph")
    parser.add_argument("--method", "-method", default="thor_olson_optimized")
    parser.add_argument("--disparity", "-disp", type=int, default=16)
    parser.add_argument("--blur", "-blur", type=int, default=21)
    parser.add_argument("--opacity", "-op", type=float, default=0.5)
    parser.add_argument("--microrelief", "-mr", choices=["on", "off"], default="on")
    parser.add_argument("--geom", "-geom", choices=["linear", "logarithmic"], default="linear")

    args = parser.parse_args()
    try:
        state.view_mode = args.mode
        state.method_name = args.method
        state.disparity = args.disparity
        state.blur = args.blur
        state.opacity = args.opacity
        state.use_microrelief = args.microrelief == "on"
        state.depth_interpretation = args.geom
    except Exception as err:
        print(f"[-] Ошибка Pydantic-валидации CLI: {err}", file=sys.stderr)
        sys.exit(1)
    return args

def scan_directory(target_file):
    file_path = os.path.abspath(target_file)
    folder = os.path.dirname(file_path)
    valid_ext = (".jpg", ".jpeg", ".png")
    try:
        files_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_ext)]
        files_list.sort(key=lambda x: x.lower())
    except Exception:
        files_list = [file_path]
    idx = files_list.index(file_path) if file_path in files_list else 0
    return files_list, idx

def main():
    args = parse_cli_args()
    all_files, file_index = scan_directory(args.file)

    try:
        # 1. Инициализация SDL / Pygame
        pygame.init()
        pygame.display.init()  # Явная пре-инициализация дисплея для корректного замера

        # Честный замер физической матрицы экрана через SDL дисплей-модуль
        display_info = pygame.display.Info()
        state.screen_w = display_info.current_w
        state.screen_h = display_info.current_h

        # Создаем hardware-accelerated полноэкранный стек (без рамок и полос)
        # Исправлено: удален несуществующий метод set_surface_full_screen()
        flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
        screen = pygame.display.set_mode((state.screen_w, state.screen_h), flags)
        pygame.display.set_caption("Dynamic Depth SDL Viewer")
        
        # Скрываем системный курсор внутри окна
        pygame.mouse.set_visible(False)

        # 2. Инициализация вычислительных конвейеров
        dp = GDepthProcessor(all_files[file_index])
        state.has_depth = dp.has_depth
        ap = AnaglyphProcessor(dp)
        renderer = VisualRenderer(state, ap)

        img_h, img_w, _ = dp.primary_img.shape
        state.scale = renderer.get_fit_to_screen_scale(img_w, img_h)

        # 3. Контроллер
        controller = KeyboardController(all_files, file_index, ap, state, renderer)

        print("[+] SDL Движок инициализирован успешно. Ореолы и белые полосы устранены.")

        clock = pygame.time.Clock()
        image_cached = None

        # 4. Главный рендер-цикл
        running = True
        while running:
            # Честный расчет дельты времени для плавного апдейта интерфейсов pygame_gui
            time_delta = clock.tick(60) / 1000.0

            if state.need_update or image_cached is None:
                image_cached = controller.update_processors_and_render()
                state.need_update = False

            # Генерируем CV2-кадр (компоновка заблюренного бэкграунда и OSD)
            display_frame = renderer.render_canvas(
                image_cached,
                os.path.basename(controller.get_current_file()),
                len(controller.files),
                controller.file_index,
                controller.ap,
            )

            # --- КОНВЕРТАЦИЯ: OpenCV (BGR) -> SDL/Pygame (RGB) ---
            frame_rgb = display_frame[:, :, ::-1]
            frame_transposed = np.transpose(frame_rgb, (1, 0, 2))
            
            # Быстрое создание SDL-поверхности напрямую из RAM-буфера numpy массива
            sdl_surface = pygame.surfarray.make_surface(frame_transposed)
            
            # Блиттинг на экран
            screen.blit(sdl_surface, (0, 0))

            # === ОБНОВЛЕНИЕ И ОТРИСОВКА ИНТЕРФЕЙСА ===
            controller.ui.update(time_delta)      # Принудительно оживляем элементы GUI
            controller.draw_ui_and_progress(screen) # Отрисовываем окно или прогресс-бар

            # Вывод кадра из буфера на дисплей
            pygame.display.flip()

            # Обрабатываем ввод и таймеры
            if not controller.handle_input(image_cached):
                running = False

        pygame.quit()
        sys.exit(0)

    except Exception as e:
        print(f"[-] Критическая ошибка SDL-сессии: {e}", file=sys.stderr)
        pygame.quit()
        sys.exit(1)

if __name__ == "__main__":
    main()

