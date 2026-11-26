"""Deterministic 2D PNG renderer for Warehouse SceneSpec instances."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from visual_task_factory.models import (
    Doorway,
    EntityType,
    ExitMarker,
    Rect2D,
    SceneEntity,
    SceneSpec,
    VisualTheme,
    WallSegment,
    Zone,
    ZoneType,
)

ColorRGBA = tuple[int, int, int, int]
ColorRGB = tuple[int, int, int]

_DEFAULT_FONT_CANDIDATES: tuple[Path, ...] = (
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\consola.ttf"),
)


@dataclass(frozen=True)
class ThemePalette:
    """Color palette for rendering a warehouse SceneSpec."""

    canvas_bg: ColorRGB
    floor_bg: ColorRGB
    grid_line: ColorRGB
    wall_fill: ColorRGB
    door_open_accent: ColorRGB
    door_blocked_accent: ColorRGB
    exit_fill: ColorRGB
    exit_text: ColorRGB
    header_bg: ColorRGB
    header_text: ColorRGB
    footer_bg: ColorRGB
    footer_text: ColorRGB
    scale_dark: ColorRGB
    scale_light: ColorRGB
    zone_fills: dict[ZoneType, ColorRGBA]
    zone_borders: dict[ZoneType, ColorRGB]
    entity_fills: dict[EntityType, ColorRGB]
    entity_borders: dict[EntityType, ColorRGB]
    entity_text: ColorRGB


_THEMES: dict[VisualTheme, ThemePalette] = {
    VisualTheme.SAFETY_STANDARD: ThemePalette(
        canvas_bg=(242, 245, 248),
        floor_bg=(250, 251, 252),
        grid_line=(228, 233, 238),
        wall_fill=(38, 45, 56),
        door_open_accent=(34, 139, 94),
        door_blocked_accent=(200, 46, 46),
        exit_fill=(22, 163, 74),
        exit_text=(255, 255, 255),
        header_bg=(30, 41, 59),
        header_text=(248, 250, 252),
        footer_bg=(235, 240, 245),
        footer_text=(30, 41, 59),
        scale_dark=(30, 41, 59),
        scale_light=(255, 255, 255),
        zone_fills={
            ZoneType.STORAGE: (59, 130, 246, 42),
            ZoneType.LOADING: (14, 165, 233, 45),
            ZoneType.HAZARD: (245, 158, 11, 68),
            ZoneType.RESTRICTED: (220, 38, 38, 64),
            ZoneType.STAGING: (16, 185, 129, 45),
        },
        zone_borders={
            ZoneType.STORAGE: (37, 99, 235),
            ZoneType.LOADING: (2, 132, 199),
            ZoneType.HAZARD: (217, 119, 6),
            ZoneType.RESTRICTED: (185, 28, 28),
            ZoneType.STAGING: (5, 150, 105),
        },
        entity_fills={
            EntityType.PALLET_RACK: (91, 124, 153),
            EntityType.CRATE: (196, 148, 92),
            EntityType.FORKLIFT: (234, 179, 8),
            EntityType.PILLAR: (71, 85, 105),
            EntityType.CONVEYOR: (120, 132, 148),
            EntityType.ROBOT: (14, 116, 144),
        },
        entity_borders={
            EntityType.PALLET_RACK: (45, 68, 90),
            EntityType.CRATE: (120, 82, 38),
            EntityType.FORKLIFT: (133, 77, 14),
            EntityType.PILLAR: (30, 41, 59),
            EntityType.CONVEYOR: (51, 65, 85),
            EntityType.ROBOT: (22, 78, 99),
        },
        entity_text=(255, 255, 255),
    ),
    VisualTheme.BLUEPRINT: ThemePalette(
        canvas_bg=(16, 38, 72),
        floor_bg=(20, 50, 94),
        grid_line=(38, 76, 132),
        wall_fill=(232, 242, 255),
        door_open_accent=(110, 231, 183),
        door_blocked_accent=(252, 165, 165),
        exit_fill=(16, 185, 129),
        exit_text=(15, 23, 42),
        header_bg=(12, 28, 56),
        header_text=(226, 238, 255),
        footer_bg=(14, 34, 66),
        footer_text=(226, 238, 255),
        scale_dark=(226, 238, 255),
        scale_light=(20, 50, 94),
        zone_fills={
            ZoneType.STORAGE: (147, 197, 253, 45),
            ZoneType.LOADING: (103, 232, 249, 45),
            ZoneType.HAZARD: (251, 191, 36, 68),
            ZoneType.RESTRICTED: (248, 113, 113, 68),
            ZoneType.STAGING: (110, 231, 183, 45),
        },
        zone_borders={
            ZoneType.STORAGE: (147, 197, 253),
            ZoneType.LOADING: (103, 232, 249),
            ZoneType.HAZARD: (251, 191, 36),
            ZoneType.RESTRICTED: (248, 113, 113),
            ZoneType.STAGING: (110, 231, 183),
        },
        entity_fills={
            EntityType.PALLET_RACK: (59, 110, 175),
            EntityType.CRATE: (180, 140, 82),
            EntityType.FORKLIFT: (217, 169, 36),
            EntityType.PILLAR: (148, 163, 184),
            EntityType.CONVEYOR: (94, 129, 172),
            EntityType.ROBOT: (45, 155, 175),
        },
        entity_borders={
            EntityType.PALLET_RACK: (219, 234, 254),
            EntityType.CRATE: (253, 230, 138),
            EntityType.FORKLIFT: (254, 240, 138),
            EntityType.PILLAR: (241, 245, 249),
            EntityType.CONVEYOR: (203, 213, 225),
            EntityType.ROBOT: (165, 243, 252),
        },
        entity_text=(255, 255, 255),
    ),
    VisualTheme.HIGH_CONTRAST: ThemePalette(
        canvas_bg=(255, 255, 255),
        floor_bg=(255, 255, 255),
        grid_line=(220, 220, 220),
        wall_fill=(0, 0, 0),
        door_open_accent=(0, 130, 60),
        door_blocked_accent=(210, 0, 0),
        exit_fill=(0, 145, 55),
        exit_text=(255, 255, 255),
        header_bg=(0, 0, 0),
        header_text=(255, 255, 255),
        footer_bg=(242, 242, 242),
        footer_text=(0, 0, 0),
        scale_dark=(0, 0, 0),
        scale_light=(255, 255, 255),
        zone_fills={
            ZoneType.STORAGE: (0, 90, 200, 38),
            ZoneType.LOADING: (0, 150, 180, 42),
            ZoneType.HAZARD: (255, 165, 0, 72),
            ZoneType.RESTRICTED: (220, 0, 0, 70),
            ZoneType.STAGING: (0, 160, 100, 40),
        },
        zone_borders={
            ZoneType.STORAGE: (0, 70, 175),
            ZoneType.LOADING: (0, 120, 150),
            ZoneType.HAZARD: (200, 100, 0),
            ZoneType.RESTRICTED: (190, 0, 0),
            ZoneType.STAGING: (0, 120, 75),
        },
        entity_fills={
            EntityType.PALLET_RACK: (45, 85, 150),
            EntityType.CRATE: (165, 100, 25),
            EntityType.FORKLIFT: (215, 155, 0),
            EntityType.PILLAR: (40, 40, 40),
            EntityType.CONVEYOR: (95, 95, 95),
            EntityType.ROBOT: (0, 115, 145),
        },
        entity_borders={
            EntityType.PALLET_RACK: (0, 0, 0),
            EntityType.CRATE: (0, 0, 0),
            EntityType.FORKLIFT: (0, 0, 0),
            EntityType.PILLAR: (0, 0, 0),
            EntityType.CONVEYOR: (0, 0, 0),
            EntityType.ROBOT: (0, 0, 0),
        },
        entity_text=(255, 255, 255),
    ),
    VisualTheme.CLUTTERED_OPS: ThemePalette(
        canvas_bg=(232, 229, 222),
        floor_bg=(241, 238, 231),
        grid_line=(214, 210, 200),
        wall_fill=(54, 49, 44),
        door_open_accent=(42, 130, 88),
        door_blocked_accent=(190, 52, 42),
        exit_fill=(32, 142, 74),
        exit_text=(255, 255, 255),
        header_bg=(54, 49, 44),
        header_text=(248, 245, 238),
        footer_bg=(224, 220, 212),
        footer_text=(42, 38, 34),
        scale_dark=(42, 38, 34),
        scale_light=(248, 245, 238),
        zone_fills={
            ZoneType.STORAGE: (78, 121, 167, 50),
            ZoneType.LOADING: (89, 161, 169, 50),
            ZoneType.HAZARD: (237, 137, 54, 72),
            ZoneType.RESTRICTED: (197, 48, 48, 68),
            ZoneType.STAGING: (72, 152, 112, 50),
        },
        zone_borders={
            ZoneType.STORAGE: (52, 92, 136),
            ZoneType.LOADING: (56, 122, 130),
            ZoneType.HAZARD: (192, 86, 33),
            ZoneType.RESTRICTED: (155, 28, 28),
            ZoneType.STAGING: (47, 118, 84),
        },
        entity_fills={
            EntityType.PALLET_RACK: (102, 114, 128),
            EntityType.CRATE: (182, 132, 76),
            EntityType.FORKLIFT: (218, 165, 24),
            EntityType.PILLAR: (82, 78, 72),
            EntityType.CONVEYOR: (118, 114, 108),
            EntityType.ROBOT: (36, 118, 134),
        },
        entity_borders={
            EntityType.PALLET_RACK: (48, 56, 66),
            EntityType.CRATE: (110, 74, 34),
            EntityType.FORKLIFT: (128, 82, 12),
            EntityType.PILLAR: (36, 34, 32),
            EntityType.CONVEYOR: (58, 54, 50),
            EntityType.ROBOT: (20, 72, 84),
        },
        entity_text=(255, 255, 255),
    ),
}


@dataclass(frozen=True)
class RenderConfig:
    """Configuration controlling deterministic image geometry and overlays."""

    pixels_per_unit: int = 24
    supersample: int = 2
    margin_x: int = 56
    margin_y: int = 30
    header_height: int = 58
    footer_height: int = 104
    audit_overlay: bool = False

    def __post_init__(self) -> None:
        if self.pixels_per_unit < 8 or self.pixels_per_unit > 96:
            raise ValueError("pixels_per_unit must be between 8 and 96.")
        if self.supersample not in (1, 2, 3):
            raise ValueError("supersample must be 1, 2, or 3.")
        if self.margin_x < 24 or self.margin_y < 16 or self.header_height < 36 or self.footer_height < 72:
            raise ValueError("Margins/header/footer sizes are too small for legible labels.")


@dataclass(frozen=True)
class RenderResult:
    """Metadata returned after rendering a SceneSpec to disk."""

    output_path: Path
    width_px: int
    height_px: int
    scene_id: str
    theme: VisualTheme
    orientation_deg: Literal[0, 90, 180, 270]
    audit_overlay: bool
    font_name: str


@dataclass(frozen=True)
class _FontBundle:
    font_name: str
    title: ImageFont.ImageFont | ImageFont.FreeTypeFont
    label: ImageFont.ImageFont | ImageFont.FreeTypeFont
    small: ImageFont.ImageFont | ImageFont.FreeTypeFont
    tiny: ImageFont.ImageFont | ImageFont.FreeTypeFont


def _load_deterministic_fonts(scale: int) -> _FontBundle:
    """Resolve fonts deterministically using fixed Windows font paths with Pillow fallback."""
    for candidate in _DEFAULT_FONT_CANDIDATES:
        if candidate.is_file():
            return _FontBundle(
                font_name=candidate.name,
                title=ImageFont.truetype(str(candidate), size=16 * scale),
                label=ImageFont.truetype(str(candidate), size=12 * scale),
                small=ImageFont.truetype(str(candidate), size=10 * scale),
                tiny=ImageFont.truetype(str(candidate), size=9 * scale),
            )

    try:
        return _FontBundle(
            font_name="pillow_default_scalable",
            title=ImageFont.load_default(size=16 * scale),
            label=ImageFont.load_default(size=12 * scale),
            small=ImageFont.load_default(size=10 * scale),
            tiny=ImageFont.load_default(size=9 * scale),
        )
    except TypeError:
        default_font = ImageFont.load_default()
        return _FontBundle(
            font_name="pillow_bitmap_default",
            title=default_font,
            label=default_font,
            small=default_font,
            tiny=default_font,
        )


def _clamp_box_to_canvas(
    x0: float,
    y0: float,
    box_w: float,
    box_h: float,
    canvas_w: int,
    canvas_h: int,
    pad: int,
) -> tuple[float, float, float, float]:
    clamped_x0 = max(float(pad), min(x0, float(canvas_w - pad) - box_w))
    clamped_y0 = max(float(pad), min(y0, float(canvas_h - pad) - box_h))
    return (clamped_x0, clamped_y0, clamped_x0 + box_w, clamped_y0 + box_h)


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return (max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1]))


def _draw_pill_label(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    center_y: float,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    bg_color: ColorRGBA | ColorRGB,
    text_color: ColorRGB,
    border_color: ColorRGB | None,
    canvas_w: int,
    canvas_h: int,
    pad_x: int,
    pad_y: int,
    margin_pad: int,
    radius: int,
) -> None:
    tw, th = _text_size(draw, text, font)
    box_w = tw + 2 * pad_x
    box_h = th + 2 * pad_y
    bx0, by0, bx1, by1 = _clamp_box_to_canvas(
        center_x - box_w / 2.0,
        center_y - box_h / 2.0,
        box_w,
        box_h,
        canvas_w,
        canvas_h,
        margin_pad,
    )
    draw.rounded_rectangle(
        (bx0, by0, bx1, by1),
        radius=radius,
        fill=bg_color,
        outline=border_color,
        width=max(1, radius // 3),
    )
    draw.text(
        ((bx0 + bx1) / 2.0, (by0 + by1) / 2.0),
        text,
        fill=text_color,
        font=font,
        anchor="mm",
    )


class WarehouseSceneRenderer:
    """Renders valid SceneSpec instances to crisp, deterministic PNG floorplans."""

    def __init__(self, config: RenderConfig | None = None) -> None:
        self.config = config or RenderConfig()

    def render(self, scene: SceneSpec, output_path: Path | str) -> RenderResult:
        """Render a SceneSpec into a PNG file at output_path."""
        if not isinstance(scene, SceneSpec):
            raise TypeError(f"Expected SceneSpec instance, got {type(scene).__name__}.")

        resolved_path = self._validate_output_path(output_path)
        cfg = self.config
        s = cfg.supersample
        ppu = cfg.pixels_per_unit * s
        margin_x = cfg.margin_x * s
        margin_y = cfg.margin_y * s
        header_h = cfg.header_height * s
        footer_h = cfg.footer_height * s
        world_y0 = header_h + margin_y

        world_w_px = int(round(scene.width * ppu))
        world_h_px = int(round(scene.height * ppu))
        canvas_w = world_w_px + 2 * margin_x
        canvas_h = header_h + margin_y + world_h_px + margin_y + footer_h
        footer_y0 = canvas_h - footer_h

        palette = _THEMES[scene.variation.theme]
        fonts = _load_deterministic_fonts(scale=s)

        def to_px(wx: float, wy: float) -> tuple[float, float]:
            return (margin_x + wx * ppu, world_y0 + wy * ppu)

        def rect_to_px(rect: Rect2D) -> tuple[float, float, float, float]:
            x0, y0 = to_px(rect.x, rect.y)
            x1, y1 = to_px(rect.x2, rect.y2)
            return (x0, y0, x1, y1)

        base_img = Image.new("RGBA", (canvas_w, canvas_h), (*palette.canvas_bg, 255))
        draw = ImageDraw.Draw(base_img)

        # 1. Header band (scene identifier & orientation compass, never overlapping floorplan).
        self._draw_header(draw, scene, palette, fonts, canvas_w, header_h, s)

        # 2. Warehouse floor area + subtle coordinate grid.
        floor_box = (margin_x, world_y0, margin_x + world_w_px, world_y0 + world_h_px)
        draw.rectangle(floor_box, fill=(*palette.floor_bg, 255))
        self._draw_floor_grid(draw, scene, palette, to_px, margin_x, world_y0, world_w_px, world_h_px, s)

        # 3. Zones (translucent fill + hatching for hazard/restricted).
        zone_overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        zone_draw = ImageDraw.Draw(zone_overlay)
        for zone in scene.zones:
            self._draw_zone_fill(zone_draw, zone, palette, rect_to_px, s)
        base_img = Image.alpha_composite(base_img, zone_overlay)
        draw = ImageDraw.Draw(base_img)

        # 4. Perimeter & interior walls.
        for wall in scene.walls:
            self._draw_wall(draw, wall, palette, to_px, ppu)

        # 5. Doorways (punch wall gap if open, or show blocked barrier).
        for door in scene.doors:
            self._draw_doorway(draw, door, palette, fonts, to_px, ppu, s, cfg.audit_overlay)

        # 6. Physical entities / racks / obstacles.
        for entity in scene.entities:
            self._draw_entity(draw, entity, palette, fonts, rect_to_px, s, cfg.audit_overlay)

        # 7. Zone title badges placed in entity-free corners where possible.
        for zone in scene.zones:
            self._draw_zone_badge(draw, zone, scene.entities, palette, fonts, rect_to_px, s)

        # 8. Footer band: exact Scale Bar + Legend swatches (drawn before exits so badges never get covered).
        self._draw_footer(
            draw,
            scene,
            palette,
            fonts,
            canvas_w,
            canvas_h,
            footer_y0,
            footer_h,
            margin_x,
            ppu,
            s,
        )

        # 9. Perimeter exit openings and legible exit badges.
        for exit_marker in scene.exits:
            self._draw_exit(
                draw,
                exit_marker,
                scene,
                palette,
                fonts,
                to_px,
                ppu,
                canvas_w,
                canvas_h,
                s,
            )

        # 10. Optional audit / debug overlay (off by default; uses only SceneSpec metadata).
        if cfg.audit_overlay:
            self._draw_audit_overlay(draw, scene, fonts, to_px, rect_to_px, margin_x, world_y0, s)

        # Downsample cleanly for crisp anti-aliased output.
        target_size = (canvas_w // s, canvas_h // s)
        final_img = base_img
        if s > 1:
            final_img = base_img.resize(target_size, resample=Image.Resampling.LANCZOS)
        rgb_img = final_img.convert("RGB")

        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            rgb_img.save(resolved_path, format="PNG", compress_level=6, optimize=False)
        except OSError as exc:
            raise OSError(f"Failed to write rendered PNG to '{resolved_path}': {exc}") from exc

        return RenderResult(
            output_path=resolved_path,
            width_px=target_size[0],
            height_px=target_size[1],
            scene_id=scene.scene_id,
            theme=scene.variation.theme,
            orientation_deg=scene.variation.orientation_deg,
            audit_overlay=cfg.audit_overlay,
            font_name=fonts.font_name,
        )

    def _validate_output_path(self, output_path: Path | str) -> Path:
        if isinstance(output_path, str):
            if not output_path.strip():
                raise ValueError("output_path must not be empty.")
            path = Path(output_path)
        elif isinstance(output_path, Path):
            if not str(output_path).strip():
                raise ValueError("output_path must not be empty.")
            path = output_path
        else:
            raise TypeError(f"output_path must be Path or str, got {type(output_path).__name__}.")

        if path.exists() and path.is_dir():
            raise IsADirectoryError(f"output_path points to an existing directory: '{path}'.")
        if path.suffix.lower() != ".png":
            raise ValueError(f"output_path must have a '.png' extension; got '{path}'.")
        return path

    def _draw_header(
        self,
        draw: ImageDraw.ImageDraw,
        scene: SceneSpec,
        palette: ThemePalette,
        fonts: _FontBundle,
        canvas_w: int,
        header_h: int,
        s: int,
    ) -> None:
        draw.rectangle((0, 0, canvas_w, header_h), fill=(*palette.header_bg, 255))
        title = f"Facility Floorplan  |  {scene.scene_id}"
        draw.text((20 * s, header_h // 2), title, fill=palette.header_text, font=fonts.title, anchor="lm")

        # North compass indicator rotated to match scene.variation.orientation_deg.
        cx = canvas_w - 36 * s
        cy = header_h // 2
        r = 16 * s
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=palette.header_text, width=max(1, s))
        # In screen coords (y down): 0 deg -> North is up (0, -1); 90 CW -> right (1, 0); etc.
        rad = math.radians(scene.variation.orientation_deg)
        dx = math.sin(rad)
        dy = -math.cos(rad)
        tip = (cx + dx * (r - 4 * s), cy + dy * (r - 4 * s))
        tail = (cx - dx * (r - 6 * s), cy - dy * (r - 6 * s))
        draw.line((tail, tip), fill=(250, 204, 21), width=max(2, 2 * s))
        draw.ellipse(
            (tip[0] - 3 * s, tip[1] - 3 * s, tip[0] + 3 * s, tip[1] + 3 * s),
            fill=(250, 204, 21),
        )
        draw.text(
            (cx - 28 * s, cy),
            f"N ({scene.variation.orientation_deg}°)",
            fill=palette.header_text,
            font=fonts.small,
            anchor="rm",
        )

    def _draw_floor_grid(
        self,
        draw: ImageDraw.ImageDraw,
        scene: SceneSpec,
        palette: ThemePalette,
        to_px,
        margin_x: int,
        header_h: int,
        world_w_px: int,
        world_h_px: int,
        s: int,
    ) -> None:
        grid_step = 2.0
        x = grid_step
        while x < scene.width:
            px, _ = to_px(x, 0.0)
            draw.line((px, header_h, px, header_h + world_h_px), fill=(*palette.grid_line, 255), width=max(1, s // 2))
            x += grid_step
        y = grid_step
        while y < scene.height:
            _, py = to_px(0.0, y)
            draw.line((margin_x, py, margin_x + world_w_px, py), fill=(*palette.grid_line, 255), width=max(1, s // 2))
            y += grid_step

    def _draw_zone_fill(
        self,
        draw: ImageDraw.ImageDraw,
        zone: Zone,
        palette: ThemePalette,
        rect_to_px,
        s: int,
    ) -> None:
        x0, y0, x1, y1 = rect_to_px(zone.bounds)
        fill_rgba = palette.zone_fills[zone.zone_type]
        border_rgb = palette.zone_borders[zone.zone_type]
        draw.rectangle((x0, y0, x1, y1), fill=fill_rgba, outline=(*border_rgb, 235), width=max(2, 2 * s))

        # Add clean diagonal hatching for hazard and cross-hatch for restricted zones.
        if zone.zone_type in (ZoneType.HAZARD, ZoneType.RESTRICTED):
            step = 14 * s
            stripe_color = (*border_rgb, 95)
            w = x1 - x0
            h = y1 - y0
            offset = -h
            while offset < w:
                p0_x = max(x0, x0 + offset)
                p0_y = y0 + max(0.0, p0_x - (x0 + offset))
                p1_x = min(x1, x0 + offset + h)
                p1_y = y0 + (p1_x - (x0 + offset))
                if p1_x > p0_x and p1_y > p0_y:
                    draw.line((p0_x, p0_y, p1_x, p1_y), fill=stripe_color, width=max(1, s))
                offset += step
            if zone.zone_type == ZoneType.RESTRICTED:
                offset = 0.0
                while offset < w + h:
                    p0_x = max(x0, x0 + offset - h)
                    p0_y = min(y1, y0 + offset - (p0_x - x0))
                    p1_x = min(x1, x0 + offset)
                    p1_y = max(y0, y0 + offset - (p1_x - x0))
                    if p1_x > p0_x and p0_y > p1_y:
                        draw.line((p0_x, p0_y, p1_x, p1_y), fill=stripe_color, width=max(1, s))
                    offset += step

    def _draw_zone_badge(
        self,
        draw: ImageDraw.ImageDraw,
        zone: Zone,
        entities: list[SceneEntity],
        palette: ThemePalette,
        fonts: _FontBundle,
        rect_to_px,
        s: int,
    ) -> None:
        x0, y0, x1, y1 = rect_to_px(zone.bounds)
        border_rgb = palette.zone_borders[zone.zone_type]
        tw, th = _text_size(draw, zone.label, fonts.small)
        pad_x = 5 * s
        pad_y = 3 * s
        badge_w = min(max(1.0, (x1 - x0) - 8 * s), tw + 2 * pad_x)
        badge_h = min(max(1.0, (y1 - y0) - 8 * s), th + 2 * pad_y)
        if badge_w < 12 * s or badge_h < 8 * s:
            return

        entity_boxes = [rect_to_px(ent.bounds) for ent in entities]
        candidate_corners = [
            (x0 + 4 * s, y0 + 4 * s, x0 + 4 * s + badge_w, y0 + 4 * s + badge_h),
            (x0 + 4 * s, y1 - 4 * s - badge_h, x0 + 4 * s + badge_w, y1 - 4 * s),
            (x1 - 4 * s - badge_w, y0 + 4 * s, x1 - 4 * s, y0 + 4 * s + badge_h),
            (x1 - 4 * s - badge_w, y1 - 4 * s - badge_h, x1 - 4 * s, y1 - 4 * s),
        ]

        chosen = candidate_corners[0]
        for cand in candidate_corners:
            if not any(
                cand[0] < eb[2] and cand[2] > eb[0] and cand[1] < eb[3] and cand[3] > eb[1]
                for eb in entity_boxes
            ):
                chosen = cand
                break

        bx0, by0, bx1, by1 = chosen
        draw.rounded_rectangle(
            (bx0, by0, bx1, by1),
            radius=3 * s,
            fill=(*border_rgb, 235),
        )
        draw.text(
            ((bx0 + bx1) / 2.0, (by0 + by1) / 2.0),
            zone.label,
            fill=(255, 255, 255, 255),
            font=fonts.small,
            anchor="mm",
        )

    def _draw_wall(
        self,
        draw: ImageDraw.ImageDraw,
        wall: WallSegment,
        palette: ThemePalette,
        to_px,
        ppu: int,
    ) -> None:
        p0 = to_px(wall.start.x, wall.start.y)
        p1 = to_px(wall.end.x, wall.end.y)
        w_px = max(3, int(round(wall.thickness * ppu)))
        draw.line((p0, p1), fill=(*palette.wall_fill, 255), width=w_px)
        half = w_px / 2.0
        for px, py in (p0, p1):
            draw.rectangle((px - half, py - half, px + half, py + half), fill=(*palette.wall_fill, 255))

    def _draw_doorway(
        self,
        draw: ImageDraw.ImageDraw,
        door: Doorway,
        palette: ThemePalette,
        fonts: _FontBundle,
        to_px,
        ppu: int,
        s: int,
        audit_overlay: bool,
    ) -> None:
        cx, cy = to_px(door.position.x, door.position.y)
        half_len = (door.width * ppu) / 2.0
        clear_half = max(6 * s, int(round(0.32 * ppu)))

        if door.orientation == "vertical":
            gap_box = (cx - clear_half, cy - half_len, cx + clear_half, cy + half_len)
            jamb1 = ((cx - clear_half, cy - half_len), (cx + clear_half, cy - half_len))
            jamb2 = ((cx - clear_half, cy + half_len), (cx + clear_half, cy + half_len))
            center_line = ((cx, cy - half_len), (cx, cy + half_len))
        else:
            gap_box = (cx - half_len, cy - clear_half, cx + half_len, cy + clear_half)
            jamb1 = ((cx - half_len, cy - clear_half), (cx - half_len, cy + clear_half))
            jamb2 = ((cx + half_len, cy - clear_half), (cx + half_len, cy + clear_half))
            center_line = ((cx - half_len, cy), (cx + half_len, cy))

        # Carve opening through the host wall.
        draw.rectangle(gap_box, fill=(*palette.floor_bg, 255))
        draw.line(jamb1[0] + jamb1[1], fill=(*palette.wall_fill, 255), width=max(2, 2 * s))
        draw.line(jamb2[0] + jamb2[1], fill=(*palette.wall_fill, 255), width=max(2, 2 * s))

        if door.is_blocked:
            accent = (*palette.door_blocked_accent, 255)
            draw.rectangle(gap_box, fill=(*palette.door_blocked_accent, 70), outline=accent, width=max(2, 2 * s))
            draw.line((gap_box[0], gap_box[1], gap_box[2], gap_box[3]), fill=accent, width=max(2, 2 * s))
            draw.line((gap_box[0], gap_box[3], gap_box[2], gap_box[1]), fill=accent, width=max(2, 2 * s))
        else:
            accent = (*palette.door_open_accent, 255)
            draw.line(center_line[0] + center_line[1], fill=accent, width=max(2, 2 * s))
            # Subtle swing arc indicating passable doorway.
            arc_r = half_len * 0.85
            if door.orientation == "vertical":
                draw.arc(
                    (cx - arc_r, cy - half_len, cx + arc_r, cy - half_len + 2 * arc_r),
                    start=0,
                    end=90,
                    fill=accent,
                    width=max(1, s),
                )
            else:
                draw.arc(
                    (cx - half_len, cy - arc_r, cx - half_len + 2 * arc_r, cy + arc_r),
                    start=270,
                    end=360,
                    fill=accent,
                    width=max(1, s),
                )

        if audit_overlay:
            draw.text((cx + 8 * s, cy - 8 * s), door.id, fill=(180, 0, 120), font=fonts.tiny, anchor="lm")

    def _draw_entity(
        self,
        draw: ImageDraw.ImageDraw,
        entity: SceneEntity,
        palette: ThemePalette,
        fonts: _FontBundle,
        rect_to_px,
        s: int,
        audit_overlay: bool,
    ) -> None:
        x0, y0, x1, y1 = rect_to_px(entity.bounds)
        fill = (*palette.entity_fills[entity.entity_type], 255)
        border = (*palette.entity_borders[entity.entity_type], 255)
        stroke_w = max(2, 2 * s)

        if entity.entity_type == EntityType.ROBOT:
            draw.rounded_rectangle((x0, y0, x1, y1), radius=6 * s, fill=fill, outline=border, width=stroke_w)
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            r = min(x1 - x0, y1 - y0) * 0.36
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 255, 255, 115), width=max(1, s))
        else:
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=border, width=stroke_w)

        if entity.entity_type == EntityType.PALLET_RACK:
            # Internal rack shelf dividers.
            w = x1 - x0
            h = y1 - y0
            if w >= h:
                for frac in (0.33, 0.67):
                    sx = x0 + w * frac
                    draw.line((sx, y0 + 2 * s, sx, y1 - 2 * s), fill=border, width=max(1, s))
            else:
                for frac in (0.33, 0.67):
                    sy = y0 + h * frac
                    draw.line((x0 + 2 * s, sy, x1 - 2 * s, sy), fill=border, width=max(1, s))

        elif entity.entity_type == EntityType.CRATE:
            pad = 3 * s
            draw.line((x0 + pad, y0 + pad, x1 - pad, y1 - pad), fill=border, width=max(1, s))
            draw.line((x0 + pad, y1 - pad, x1 - pad, y0 + pad), fill=border, width=max(1, s))

        elif entity.entity_type == EntityType.PILLAR:
            draw.line((x0, y0, x1, y1), fill=border, width=max(2, s))
            draw.line((x0, y1, x1, y0), fill=border, width=max(2, s))

        elif entity.entity_type == EntityType.CONVEYOR:
            w = x1 - x0
            h = y1 - y0
            step = 10 * s
            if w >= h:
                cx_cur = x0 + step
                while cx_cur < x1 - 3 * s:
                    draw.line((cx_cur, y0 + 3 * s, cx_cur, y1 - 3 * s), fill=border, width=max(1, s))
                    cx_cur += step
            else:
                cy_cur = y0 + step
                while cy_cur < y1 - 3 * s:
                    draw.line((x0 + 3 * s, cy_cur, x1 - 3 * s, cy_cur), fill=border, width=max(1, s))
                    cy_cur += step

        elif entity.entity_type == EntityType.FORKLIFT:
            # Draw two dark fork tines inside the front end of the forklift footprint.
            w = x1 - x0
            h = y1 - y0
            if w >= h:
                fork_x0 = x1 - w * 0.28
                draw.line((fork_x0, y0 + h * 0.28, x1 - 2 * s, y0 + h * 0.28), fill=border, width=max(2, 2 * s))
                draw.line((fork_x0, y0 + h * 0.72, x1 - 2 * s, y0 + h * 0.72), fill=border, width=max(2, 2 * s))
            else:
                fork_y0 = y1 - h * 0.28
                draw.line((x0 + w * 0.28, fork_y0, x0 + w * 0.28, y1 - 2 * s), fill=border, width=max(2, 2 * s))
                draw.line((x0 + w * 0.72, fork_y0, x0 + w * 0.72, y1 - 2 * s), fill=border, width=max(2, 2 * s))

        # Fit readable label inside entity box when large enough, otherwise skip cluttering tiny pillars.
        box_w = x1 - x0
        box_h = y1 - y0
        label_text = entity.id if audit_overlay else entity.label
        tw, th = _text_size(draw, label_text, fonts.tiny)
        if box_w >= tw + 6 * s and box_h >= th + 4 * s:
            draw.text(
                ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                label_text,
                fill=palette.entity_text,
                font=fonts.tiny,
                anchor="mm",
            )

    def _draw_exit(
        self,
        draw: ImageDraw.ImageDraw,
        exit_marker: ExitMarker,
        scene: SceneSpec,
        palette: ThemePalette,
        fonts: _FontBundle,
        to_px,
        ppu: int,
        canvas_w: int,
        canvas_h: int,
        s: int,
    ) -> None:
        cx, cy = to_px(exit_marker.position.x, exit_marker.position.y)
        half_w = (exit_marker.width * ppu) / 2.0
        clear_half = max(6 * s, int(round(0.32 * ppu)))

        # Determine nearest perimeter edge for opening orientation & outward badge offset.
        dist_w = exit_marker.position.x
        dist_e = scene.width - exit_marker.position.x
        dist_n = exit_marker.position.y
        dist_s = scene.height - exit_marker.position.y
        nearest = min(
            [("W", dist_w), ("E", dist_e), ("N", dist_n), ("S", dist_s)],
            key=lambda item: item[1],
        )[0]

        if nearest in ("N", "S"):
            gap_box = (cx - half_w, cy - clear_half, cx + half_w, cy + clear_half)
            badge_x = cx
            badge_y = cy - 22 * s if nearest == "N" else cy + 22 * s
        else:
            gap_box = (cx - clear_half, cy - half_w, cx + clear_half, cy + half_w)
            badge_x = cx - 26 * s if nearest == "W" else cx + 26 * s
            badge_y = cy

        draw.rectangle(gap_box, fill=(*palette.floor_bg, 255))
        draw.rectangle(
            gap_box,
            fill=(*palette.exit_fill, 95),
            outline=(*palette.exit_fill, 255),
            width=max(2, 2 * s),
        )

        _draw_pill_label(
            draw=draw,
            center_x=badge_x,
            center_y=badge_y,
            text=exit_marker.id,
            font=fonts.label,
            bg_color=(*palette.exit_fill, 255),
            text_color=palette.exit_text,
            border_color=(255, 255, 255),
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            pad_x=7 * s,
            pad_y=3 * s,
            margin_pad=6 * s,
            radius=5 * s,
        )

    def _draw_audit_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        scene: SceneSpec,
        fonts: _FontBundle,
        to_px,
        rect_to_px,
        margin_x: int,
        header_h: int,
        s: int,
    ) -> None:
        # Coordinate ticks along top and left perimeter of world area.
        tick_step = 5.0
        wx = 5.0
        while wx <= scene.width + 1e-6:
            px, _ = to_px(wx, 0.0)
            draw.line((px, header_h - 4 * s, px, header_h + 4 * s), fill=(220, 38, 127), width=max(1, s))
            draw.text((px, header_h - 6 * s), f"{wx:g}", fill=(220, 38, 127), font=fonts.tiny, anchor="mb")
            wx += tick_step

        wy = 0.0
        while wy <= scene.height + 1e-6:
            _, py = to_px(0.0, wy)
            draw.line((margin_x - 4 * s, py, margin_x + 4 * s, py), fill=(220, 38, 127), width=max(1, s))
            draw.text((margin_x - 6 * s, py), f"{wy:g}", fill=(220, 38, 127), font=fonts.tiny, anchor="rm")
            wy += tick_step

        for zone in scene.zones:
            x0, y0, x1, y1 = rect_to_px(zone.bounds)
            draw.text((x1 - 4 * s, y1 - 4 * s), zone.id, fill=(190, 24, 93), font=fonts.tiny, anchor="rb")

    def _draw_footer(
        self,
        draw: ImageDraw.ImageDraw,
        scene: SceneSpec,
        palette: ThemePalette,
        fonts: _FontBundle,
        canvas_w: int,
        canvas_h: int,
        footer_y0: int,
        footer_h: int,
        margin_x: int,
        ppu: int,
        s: int,
    ) -> None:
        draw.rectangle((0, footer_y0, canvas_w, canvas_h), fill=(*palette.footer_bg, 255))
        draw.line((0, footer_y0, canvas_w, footer_y0), fill=(*palette.wall_fill, 90), width=max(1, s))

        # Left side: exact Scale Bar.
        sb_x0 = margin_x
        sb_y0 = footer_y0 + 28 * s
        sb_len = max(20.0, scene.scale_bar.world_units * ppu)
        sb_h = 10 * s
        segments = 4
        seg_w = sb_len / segments
        for idx in range(segments):
            sx0 = sb_x0 + idx * seg_w
            sx1 = sx0 + seg_w
            fill = palette.scale_dark if idx % 2 == 0 else palette.scale_light
            draw.rectangle(
                (sx0, sb_y0, sx1, sb_y0 + sb_h),
                fill=(*fill, 255),
                outline=(*palette.scale_dark, 255),
                width=max(1, s),
            )

        draw.text((sb_x0, sb_y0 - 6 * s), "0", fill=palette.footer_text, font=fonts.small, anchor="lb")
        mid_m = scene.scale_bar.display_meters / 2.0
        draw.text(
            (sb_x0 + sb_len / 2.0, sb_y0 - 6 * s),
            f"{mid_m:g} {scene.scale_bar.unit_label}",
            fill=palette.footer_text,
            font=fonts.small,
            anchor="mb",
        )
        draw.text(
            (sb_x0 + sb_len, sb_y0 - 6 * s),
            f"{scene.scale_bar.display_meters:g} {scene.scale_bar.unit_label}",
            fill=palette.footer_text,
            font=fonts.small,
            anchor="rb",
        )
        draw.text(
            (sb_x0, sb_y0 + sb_h + 10 * s),
            f"Scale Bar ({scene.scale_bar.world_units:g} units = {scene.scale_bar.display_meters:g} {scene.scale_bar.unit_label})",
            fill=palette.footer_text,
            font=fonts.tiny,
            anchor="lt",
        )

        # Right side: compact visual legend.
        legend_items: list[tuple[str, ColorRGB]] = [
            ("Storage", palette.zone_borders[ZoneType.STORAGE]),
            ("Hazard", palette.zone_borders[ZoneType.HAZARD]),
            ("Restricted", palette.zone_borders[ZoneType.RESTRICTED]),
            ("Exit", palette.exit_fill),
            ("Open Door", palette.door_open_accent),
            ("Blocked Door", palette.door_blocked_accent),
            ("Rack / Obst.", palette.entity_fills[EntityType.PALLET_RACK]),
        ]
        lx_start = max(sb_x0 + sb_len + 36 * s, canvas_w * 0.34)
        col_w = max(95 * s, (canvas_w - margin_x - lx_start) / 4.0)
        for idx, (label, swatch_color) in enumerate(legend_items):
            col = idx % 4
            row = idx // 4
            ix = lx_start + col * col_w
            iy = footer_y0 + (24 + row * 30) * s
            if ix + 75 * s > canvas_w - 8 * s:
                continue
            box_sz = 12 * s
            draw.rectangle(
                (ix, iy - box_sz / 2.0, ix + box_sz, iy + box_sz / 2.0),
                fill=(*swatch_color, 255),
                outline=(*palette.footer_text, 200),
                width=max(1, s),
            )
            draw.text(
                (ix + box_sz + 6 * s, iy),
                label,
                fill=palette.footer_text,
                font=fonts.small,
                anchor="lm",
            )


def render_scene(
    scene: SceneSpec,
    output_path: Path | str,
    *,
    config: RenderConfig | None = None,
    audit_overlay: bool = False,
) -> RenderResult:
    """Render a SceneSpec into a deterministic PNG floorplan image."""
    effective_config = config
    if effective_config is None:
        effective_config = RenderConfig(audit_overlay=audit_overlay)
    elif audit_overlay and not effective_config.audit_overlay:
        effective_config = RenderConfig(
            pixels_per_unit=effective_config.pixels_per_unit,
            supersample=effective_config.supersample,
            margin_x=effective_config.margin_x,
            header_height=effective_config.header_height,
            footer_height=effective_config.footer_height,
            audit_overlay=True,
        )
    return WarehouseSceneRenderer(config=effective_config).render(
        scene=scene,
        output_path=output_path,
    )
