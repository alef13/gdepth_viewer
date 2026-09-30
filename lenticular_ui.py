# -*- coding: utf-8 -*-
"""Модуль графического интерфейса пользователя (GUI) для настройки лентикулярной печати."""

import pygame
import pygame_gui


class LenticularUI(object):
    """Класс, полностью отвечающий за отрисовку диалоговых окон параметров

    лентикулярной печати и рендеринг прогресс-бара генерации.
    """

    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.manager = pygame_gui.UIManager((screen_w, screen_h))
        self.window = None
        
        # Переменные состояния прогресса
        self.progress_val = 0.0
        self.progress_text = ""
        self.is_generating = False

        # Элементы формы
        self.input_lpi = None
        self.input_dpi = None
        self.input_width = None
        self.input_views = None
        self.input_shift = None
        self.btn_cancel = None
        self.btn_start = None

    def show_dialog(self):
        """Строит и позиционирует центрированное окно с полями настроек растра."""
        pygame.mouse.set_visible(True)
        self.is_generating = False

        win_w, win_h = 450, 380
        start_x = (self.screen_w - win_w) // 2
        start_y = (self.screen_h - win_h) // 2

        self.window = pygame_gui.elements.UIWindow(
            rect=pygame.Rect((start_x, start_y), (win_w, win_h)),
            manager=self.manager,
            window_display_title="Lenticular Interlacing Settings"
        )

        # Текстовые метки и поля ввода данных (UITextInputLine заменен на UITextEntryLine)
        pygame_gui.elements.UILabel(relative_rect=pygame.Rect((20, 20), (160, 25)), text="Lens Pitch (LPI):", manager=self.manager, container=self.window)
        self.input_lpi = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 20), (200, 30)), manager=self.manager, container=self.window, initial_text="75.24")

        pygame_gui.elements.UILabel(relative_rect=pygame.Rect((20, 60), (160, 25)), text="Printer DPI:", manager=self.manager, container=self.window)
        self.input_dpi = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 60), (200, 30)), manager=self.manager, container=self.window, initial_text="1200")

        pygame_gui.elements.UILabel(relative_rect=pygame.Rect((20, 100), (160, 25)), text="Print Width (Inches):", manager=self.manager, container=self.window)
        self.input_width = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 100), (200, 30)), manager=self.manager, container=self.window, initial_text="10.0")

        pygame_gui.elements.UILabel(relative_rect=pygame.Rect((20, 140), (160, 25)), text="Number of Views:", manager=self.manager, container=self.window)
        self.input_views = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 140), (200, 30)), manager=self.manager, container=self.window, initial_text="16")

        pygame_gui.elements.UILabel(relative_rect=pygame.Rect((20, 180), (160, 25)), text="Stereo Base (Max Shift):", manager=self.manager, container=self.window)
        self.input_shift = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((200, 180), (200, 30)), manager=self.manager, container=self.window, initial_text="32")

        # Кнопки Действия
        self.btn_cancel = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((40, 260), (160, 40)), text="Cancel", manager=self.manager, container=self.window)
        self.btn_start = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((230, 260), (160, 40)), text="GENERATE", manager=self.manager, container=self.window)

    def close_dialog(self):
        """Безопасно уничтожает окно GUI."""
        if self.window:
            self.window.kill()
            self.window = None
        pygame.mouse.set_visible(False)

    def get_parsed_params(self, src_img_copy):
        """Собирает текстовые данные из инпутов и приводит их к нужным типам."""
        return {
            "lpi": float(self.input_lpi.get_text()),
            "dpi": int(self.input_dpi.get_text()),
            "print_w_inch": float(self.input_width.get_text()),
            "views_count": int(self.input_views.get_text()),
            "max_shift": int(self.input_shift.get_text()),
            "src_image": src_img_copy
        }

    def update(self, time_delta):
        """Обновляет внутренние анимации элементов управления pygame_gui."""
        if self.window:
            self.manager.update(time_delta)

    def draw(self, screen_surface):
        """Рендерит активные элементы GUI либо полноэкранный ProgressBar расчета."""
        if self.window:
            self.manager.draw_ui(screen_surface)
            
        elif self.is_generating:
            # Магия отрисовки статус-бара поверх затемненного экрана
            overlay = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            screen_surface.blit(overlay, (0, 0))

            bar_w, bar_h = 600, 35
            bx = (self.screen_w - bar_w) // 2
            by = (self.screen_h - bar_h) // 2

            font = pygame.font.SysFont("Arial", 20, bold=True)
            txt_surf = font.render(f"LENTICULAR GENERATION: {self.progress_text}", True, (255, 255, 255))
            screen_surface.blit(txt_surf, (bx, by - 35))

            pygame.draw.rect(screen_surface, (50, 50, 50), (bx, by, bar_w, bar_h), 0, 4)
            fill_w = int(bar_w * self.progress_val)
            if fill_w > 0:
                pygame.draw.rect(screen_surface, (0, 255, 128), (bx, by, fill_w, bar_h), 0, 4)
            pygame.draw.rect(screen_surface, (200, 200, 200), (bx, by, bar_w, bar_h), 2, 4)
            
            pct_surf = font.render(f"{int(self.progress_val * 100)}%", True, (0, 0, 0) if self.progress_val > 0.5 else (255, 255, 255))
            screen_surface.blit(pct_surf, (bx + (bar_w - pct_surf.get_width()) // 2, by + 5))

