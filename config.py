# -*- coding: utf-8 -*-
"""Модуль конфигурации и валидации состояния приложения."""

from pydantic import BaseModel, Field, field_validator, PrivateAttr
from typing import List


class ViewerState(BaseModel):
    """Строго типизированная модель состояния просмотрщика на Pydantic."""

    scale: float = Field(default=1.0, description="Текущий масштаб изображения")
    offset_x: int = Field(default=0, description="Смещение по оси X")
    offset_y: int = Field(default=0, description="Смещение по оси Y")

    disparity: int = Field(default=16, ge=0, le=100, description="Величина параллакса")
    blur: int = Field(default=21, ge=3, le=101, description="Радиус размытия деталей")
    opacity: float = Field(default=0.5, ge=0.0, le=1.0, description="Сила микрорельефа")
    depth_interpretation: str = Field(
        default="linear",
        description="Тип интерпретации карты глубин (linear или logarithmic)"
    )
    convergence: int = Field(
        default=0, ge=-50, le=50, description="Сдвиг плоскости нулевого параллакса"
    )

    method_name: str = Field(default="thor_olson_optimized", description="Имя активного метода")
    view_mode: str = Field(default="anaglyph", description="Режим отображения")

    screen_w: int = Field(default=1920, gt=0, description="Ширина экрана")
    screen_h: int = Field(default=1080, gt=0, description="Высота экрана")
    need_update: bool = Field(default=True, description="Флаг необходимости пересчета")

    use_microrelief: bool = Field(default=True, description="Флаг включения микрорельефа")
    show_osd: bool = Field(default=True, description="Флаг отображения OSD")
    show_help: bool = Field(default=False, description="Флаг отображения помощи")
    has_depth: bool = Field(default=True, description="Флаг наличия 3D метаданных в активном файле")

    slideshow_active: bool = Field(default=False, description="Флаг активности слайд-шоу")
    slideshow_delay: float = Field(default=8.0, ge=0.5, le=30.0, description="Задержка слайд-шоу в секундах")
    show_slideshow_status: bool = Field(default=True, description="Показывать статус слайд-шоу")
    slideshow_order: str = Field(
        default="sequential",
        description="Порядок слайд-шоу: sequential (по очереди) или random (случайный)"
    )

    slideshow_accumulator: float = Field(default=0.0, description="Накопленное реальное время текущего слайда в секундах")

    # === НАСТРОЙКИ ИНТЕРАКТИВНОГО ПАРАЛЛАКСА (FACEBOOK 3D) ===
    fb_3d_mode: bool = Field(default=False, description="Флаг активности режима Facebook 3D")
    mouse_dx: float = Field(default=0.0, description="Текущее отклонение мыши по X (-1.0 до 1.0)")
    mouse_dy: float = Field(default=0.0, description="Текущее отклонение мыши по Y (-1.0 до 1.0)")

    # === НАСТРОЙКИ ВЕРСИИ 2 (UX, АНТИ-ГОСТИНГ, WIGGLE И ГИСТОГРАММА) ===
    anaglyph_ghost_reduction: bool = Field(default=True, description="Флаг умного подавления двоения каналов")
    last_input_time: float = Field(default=0.0, description="Системное время последнего действия пользователя")
    wiggle_phase: float = Field(default=0.0, description="Текущая временная фаза генерации колебания")
    wiggle_speed: float = Field(default=12.0, ge=2.0, le=30.0, description="Частота анимации Wiggle в Герцах")
    show_histogram: bool = Field(default=True, description="Флаг отображения аналитической гистограммы планов")

    @field_validator("blur")
    @classmethod
    def check_blur_is_odd(cls, v: int) -> int:
        """Автоматическая валидация: радиус размытия OpenCV должен быть нечетным."""
        if v % 2 == 0:
            return v + 1
        return v

