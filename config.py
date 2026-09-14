# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field, field_validator


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
    slideshow_delay: float = Field(default=4.0, ge=0.5, le=30.0, description="Задержка слайд-шоу в секундах")
    show_slideshow_status: bool = Field(default=True, description="Показывать статус слайд-шоу на экране")
    slideshow_order: str = Field(
        default="sequential",
        description="Порядок слайд-шоу: sequential (по очереди) или random (случайный)"
    )

    convergence: int = Field(
        default=0, ge=-150, le=150, description="Сдвиг плоскости нулевого параллакса"
    )


    @field_validator("blur")
    @classmethod
    def check_blur_is_odd(cls, v: int) -> int:
        if v % 2 == 0:
            return v + 1
        return v

