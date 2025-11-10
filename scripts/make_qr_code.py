import segno
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageColor
from pathlib import Path
from typing import Optional, Tuple, Union

def _load_font(size: int, fallback_paths: Optional[Tuple[Path, ...]] = None) -> ImageFont.ImageFont:
    fallback_paths = fallback_paths or ()
    for custom_path in fallback_paths:
        try:
            return ImageFont.truetype(str(custom_path), size)
        except OSError:
            continue
    for font_name in ("Poppins Bold.ttf", "Arial.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _normalize_rgb(color: Union[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if isinstance(color, str):
        return ImageColor.getrgb(color)
    if isinstance(color, (tuple, list)) and len(color) >= 3:
        return tuple(int(c) for c in color[:3])
    raise ValueError(f"Unsupported color specification: {color}")


def make_qr_with_text(
    url: str,
    text: str,
    save_path: Path,
    subheading: Optional[str] = None,
    gradient_start: Union[str, Tuple[int, int, int]] = (205, 102, 12),
    gradient_end: Union[str, Tuple[int, int, int]] = (12, 34, 79),
    text_scale: float = 0.22 * (2 / 3),
    logo_path: Optional[Path] = None,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    qr = segno.make(url, error="h", boost_error=True)
    tmp_path = save_path.with_name(f"{save_path.stem}_raw.png")
    qr.save(tmp_path, scale=18, border=2, dark="#000000", light="#FFFFFF")
    qr_img = Image.open(tmp_path).convert("RGBA")
    width, height = qr_img.size

    gradient = Image.new("RGBA", (width, height))
    grad_draw = ImageDraw.Draw(gradient)
    start_rgb = _normalize_rgb(gradient_start)
    end_rgb = _normalize_rgb(gradient_end)
    for x in range(width):
        blend = x / max(1, width - 1)
        color = (
            int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * blend),
            int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * blend),
            int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * blend),
            255,
        )
        grad_draw.line([(x, 0), (x, height)], fill=color)
    gradient = gradient.filter(ImageFilter.GaussianBlur(radius=1.5))

    img = Image.new("RGBA", (width, height), LIGHT_COLOR)
    module_mask = ImageOps.invert(qr_img.convert("L")).point(lambda p: 255 if p > 10 else 0, mode="L")
    img.paste(gradient, mask=module_mask)
    qr_img.close()
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            pass

    # Overlay text inside the QR code
    font_size = max(12, int(img.size[0] * text_scale))
    font_path = Path("/Users/simon/Desktop/envs/simba/simba/simba/assets/fonts/Poppins Bold.ttf")
    font = _load_font(font_size, fallback_paths=[font_path])
    text_bbox = ImageDraw.Draw(img).textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    center_x = img.size[0] // 2
    center_y = img.size[1] // 2
    text_pos = (center_x - text_width // 2, center_y - text_height // 2 - int(img.size[0] * 0.05))

    shadow_offset = max(3, int(font_size * 0.12))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text(
        (text_pos[0] + shadow_offset, text_pos[1] + shadow_offset),
        text,
        font=font,
        fill=(15, 23, 42, 160),
    )
    img = Image.alpha_composite(img, shadow)

    stroke = max(2, int(font_size * 0.08))
    draw = ImageDraw.Draw(img)
    draw.text(
        text_pos,
        text,
        fill=(246, 248, 255, 255),
        font=font,
        stroke_width=stroke,
        stroke_fill="#2b1111",
    )

    if subheading:
        sub_font_size = max(8, int(font_size * 0.36))
        sub_font = _load_font(sub_font_size, fallback_paths=[font_path])
        sub_text = subheading
        sub_bbox = draw.textbbox((0, 0), sub_text, font=sub_font)
        sub_width = sub_bbox[2] - sub_bbox[0]
        sub_height = sub_bbox[3] - sub_bbox[1]
        sub_y = center_y + text_height // 2 + int(font_size * 0.12)
        sub_pos = (center_x - sub_width // 2, sub_y)

        sub_shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sub_shadow_offset = max(2, int(sub_font_size * 0.18))
        ImageDraw.Draw(sub_shadow).text(
            (sub_pos[0] + sub_shadow_offset, sub_pos[1] + sub_shadow_offset),
            sub_text,
            font=sub_font,
            fill=(15, 23, 42, 130),
        )
        img = Image.alpha_composite(img, sub_shadow)
        draw = ImageDraw.Draw(img)

        draw.text(
            sub_pos,
            sub_text,
            fill=LIGHT_COLOR,
            font=sub_font,
            stroke_width=max(1, int(sub_font_size * 0.12)),
            stroke_fill="#2b1111",
        )

    border = max(2, int(width * 0.013))
    outer_size = (width + border * 2, height + border * 2)
    corner_radius = max(8, int(border * 1.4))
    framed = Image.new("RGBA", outer_size, (0, 0, 0, 0))
    frame_draw = ImageDraw.Draw(framed)
    outer_rect = (0, 0, outer_size[0] - 1, outer_size[1] - 1)
    frame_draw.rounded_rectangle(outer_rect, radius=corner_radius, fill=(15, 15, 18, 255))

    inner_radius = max(6, corner_radius - border)
    inner_rect = (
        border,
        border,
        outer_size[0] - border - 1,
        outer_size[1] - border - 1,
    )
    frame_draw.rounded_rectangle(inner_rect, radius=inner_radius, fill=LIGHT_COLOR)
    framed.alpha_composite(img, (border, border))

    upscale_factor = 2
    output = framed.resize(
        (framed.width * upscale_factor, framed.height * upscale_factor),
        resample=Image.LANCZOS,
    )
    output.convert("RGB").save(save_path)



URL = "https://github.com/sgoldenlab/simba"
SAVE_PATH = Path("/Users/simon/Desktop/envs/simba/simba/misc/simba_qr_github.png")
TEXT = "SimBA"
SUBHEADING = "GitHub"
LIGHT_COLOR = "#F8FAFC"
DARK_COLOR = "#080A4D"

make_qr_with_text(
    URL,
    TEXT,
    SAVE_PATH,
    subheading=SUBHEADING,
    gradient_start="navy",
    gradient_end="black",
    text_scale=0.20 * (2 / 3),
)


# if __name__ == "__main__":
#     SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
#     make_pretty_qr(URL, SAVE_PATH)