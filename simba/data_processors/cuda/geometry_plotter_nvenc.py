__author__ = "Simon Nilsson"

"""
Overlay geometries on a video, fully on the GPU (NVDEC -> CUDA -> NVENC).

GPU counterpart of the CPU-multiprocessing :class:`~simba.plotting.geometry_plotter.GeometryPlotter`.
Per frame, filled shapes are rasterised directly onto the decoded frame by a CUDA kernel (point-in-shape
test per pixel) and alpha-blended, then NVENC-encoded.

Scope: filled **Polygon** and **Point** (circle) geometries with per-geometry colour, optional
**outlines** (``outline_clr`` / ``thickness``), and shape-/background-opacity. LineString/MultiPolygon/
MultiLineString are approximated by their exterior / convex-hull polygon. ROIs can be read directly via
:meth:`GeometryPlotterNVENC.from_roi`. NOT yet supported vs the CPU version: polygon interiors (holes)
and intersection re-colouring - use :class:`~simba.plotting.geometry_plotter.GeometryPlotter` for those.

Follows the GreyscaleNVENC / PosePlotterNVENC idiom: self-contained ``object`` subclass, module-level
kernel, sharing ``cuda/utils.py`` NVDEC/NVENC factories and the ``cuda/image.py`` geometry device
primitives ``_cuda_is_inside_polygon`` / ``_cuda_is_inside_circle``. GPU-only.
"""

import math
import os
import subprocess
import tempfile
import threading
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from numba import cuda

try:
    import torch
except Exception:
    torch = None

from shapely.geometry import (LineString, MultiLineString, MultiPolygon, Point,
                              Polygon)

from simba.data_processors.cuda.image import (_cuda_is_inside_circle,
                                              _cuda_is_inside_polygon)
from simba.data_processors.cuda.utils import (_cuda_point_to_segment_dist,
                                              get_nvc_decoder, get_nvc_encoder)
from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_if_valid_rgb_tuple,
                                check_instance, check_int,
                                check_nvidea_gpu_available, check_str,
                                check_valid_boolean, check_valid_lst)
from simba.utils.data import create_color_palettes
from simba.utils.enums import Keys
from simba.utils.errors import (FrameRangeError, InvalidInputError,
                                SimBAGPUError)
from simba.utils.lookups import get_color_dict, get_nvdec_count
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (get_fn_ext, get_pkg_version,
                                    get_video_meta_data)

NVENC_CODECS = ('h264', 'hevc', 'av1')
TPB = (16, 16)


@cuda.jit()
def _geometry_overlay_kernel(rgb, argb, poly_verts, poly_vcount, poly_bbox, circles, colors, shape_op, bg_op, outline_clr, thickness):
    """Rasterise this frame's geometries onto the ARGB buffer.

    ``poly_verts`` (n_geo, max_v, 2) int32, ``poly_vcount`` (n_geo,), ``poly_bbox`` (n_geo, 4) =
    (xmin, ymin, xmax, ymax), ``circles`` (n_geo, 3) = (cx, cy, r), ``colors`` (n_geo, 3) in B, G, R.
    A pixel inside geometry g is alpha-blended with ``colors[g]`` at ``shape_op``; the background is first
    scaled by ``bg_op`` (matching the CPU fade). When ``thickness`` > 0, pixels within ``thickness/2`` of
    any polygon edge / circle border are painted ``outline_clr`` (B, G, R) on top. The per-polygon bounding
    box cheaply skips the O(vertices) point-in-polygon / edge tests for pixels nowhere near a shape. RGB in;
    ARGB out."""
    x, y = cuda.grid(2)
    h, w = rgb.shape[0], rgb.shape[1]
    if y < h and x < w:
        opac = 1.0 - bg_op
        add = 255.0 * opac * opac
        out_r = rgb[y][x][0] * bg_op + add
        out_g = rgb[y][x][1] * bg_op + add
        out_b = rgb[y][x][2] * bg_op + add
        n_geo = poly_vcount.shape[0]
        for gi in range(n_geo):
            hit = False
            vc = poly_vcount[gi]
            if vc >= 3 and x >= poly_bbox[gi, 0] and x <= poly_bbox[gi, 2] and y >= poly_bbox[gi, 1] and y <= poly_bbox[gi, 3]:
                if _cuda_is_inside_polygon(x, y, poly_verts[gi, :vc]):
                    hit = True
            if not hit and circles[gi, 2] > 0:
                if _cuda_is_inside_circle(x, y, circles[gi, 0], circles[gi, 1], circles[gi, 2]):
                    hit = True
            if hit:
                out_b = colors[gi, 0] * shape_op + out_b * (1.0 - shape_op)
                out_g = colors[gi, 1] * shape_op + out_g * (1.0 - shape_op)
                out_r = colors[gi, 2] * shape_op + out_r * (1.0 - shape_op)
        if thickness > 0:
            half = thickness * 0.5
            on = False
            gi = 0
            while gi < n_geo and not on:
                vc = poly_vcount[gi]
                if vc >= 2 and x >= poly_bbox[gi, 0] - half and x <= poly_bbox[gi, 2] + half and y >= poly_bbox[gi, 1] - half and y <= poly_bbox[gi, 3] + half:
                    e = 0
                    while e < vc and not on:
                        nxt = e + 1
                        if nxt >= vc:
                            nxt = 0
                        if _cuda_point_to_segment_dist(x, y, poly_verts[gi, e, 0], poly_verts[gi, e, 1], poly_verts[gi, nxt, 0], poly_verts[gi, nxt, 1]) <= half:
                            on = True
                        e += 1
                if not on and circles[gi, 2] > 0:
                    ddx = x - circles[gi, 0]
                    ddy = y - circles[gi, 1]
                    if abs(math.sqrt(ddx * ddx + ddy * ddy) - circles[gi, 2]) <= half:
                        on = True
                gi += 1
            if on:
                out_b = outline_clr[0]
                out_g = outline_clr[1]
                out_r = outline_clr[2]
        argb[y][x][0] = out_b
        argb[y][x][1] = out_g
        argb[y][x][2] = out_r
        argb[y][x][3] = 255


class GeometryPlotterNVENC(object):
    """Overlay geometries on a video, fully on the GPU (NVDEC -> CUDA -> NVENC). GPU-only.

    GPU counterpart of :class:`~simba.plotting.geometry_plotter.GeometryPlotter`. See the module docstring
    for the supported-geometry scope (filled Polygon + Point, optional outlines; interiors/intersection-
    colouring are not yet implemented). ROIs can be loaded directly with :meth:`from_roi`.

    .. seealso::
       Full-featured CPU version: :class:`~simba.plotting.geometry_plotter.GeometryPlotter`.

    .. csv-table:: Measured runtime (RTX 4070, h264, threaded decode, n_workers=1)
       :header: Resolution, Frames, Geometries, Time (s), Encode FPS
       :widths: 20, 14, 22, 14, 14
       :align: center

       800x600, 108000, 2 polygons, 77.8, 1388
       1280x720, 108000, 3 ROIs (rect/circle/polygon), 110.1, 962


    Provide the geometries as exactly ONE of: ``geometries`` (a per-frame list), ``roi_path`` (a SimBA
    ``ROI_definitions.h5``), or ``config_path`` (a SimBA project). :meth:`from_roi` is a thin convenience
    wrapper around the ``roi_path`` / ``config_path`` inputs with outline-only defaults.

    :param Union[str, os.PathLike] video_path: Path to the video to overlay onto.
    :param Optional[List[List]] geometries: List (per geometry/track) of per-frame shapely geometries, as for the CPU :class:`~simba.plotting.geometry_plotter.GeometryPlotter`. Requires ``colors`` or ``palette``.
    :param Optional[Union[str, os.PathLike]] roi_path: Path to a SimBA ``ROI_definitions.h5``; its ROIs (with their own colours) are drawn. Alternative to ``geometries``.
    :param Optional[Union[str, os.PathLike]] config_path: Path to a SimBA ``project_config.ini`` to read ROIs from the project. Alternative to ``geometries`` / ``roi_path``.
    :param Optional[str] video_name: When reading ROIs, the video whose ROIs to draw. Defaults to the ``video_path`` file name.
    :param Union[str, os.PathLike] save_path: Output path. If None, ``<video>_geometry.mp4`` next to the source.
    :param Optional[List[Union[str, Tuple[int,int,int]]]] colors: One colour per geometry (BGR tuple or SimBA colour name). Used with ``geometries``; provide this or ``palette``.
    :param Optional[str] palette: Colour palette name (used with ``geometries`` if ``colors`` is None).
    :param float shape_opacity: Fill opacity 0-1 (0 = no fill, outline only). Default 0.5.
    :param float bg_opacity: Background opacity 0-1. Default 1.0.
    :param Optional[Tuple[int,int,int]] outline_clr: BGR colour for polygon/circle outlines. None = no outline. Default None.
    :param Optional[int] thickness: Outline thickness in pixels (used when ``outline_clr`` is set). Default 2.
    :param Optional[int] circle_size: Radius for Point geometries. Default: auto from frame size.
    :param str codec: NVENC codec ('h264', 'hevc', 'av1'). Default 'h264'.
    :param int buffer_size: Threaded NVDEC read-ahead per worker. Default 16.
    :param Optional[int] n_workers: Parallel decode/encode pipelines; None auto-detects NVDEC count. Default None.
    :param bool verbose: Print progress. Default True.

    :example:

    >>> # per-frame geometries:
    >>> GeometryPlotterNVENC(video_path='in.mp4', geometries=geos, colors=['Red', 'Green'], save_path='out.mp4').run()
    >>> # or straight from a SimBA ROI file:
    >>> GeometryPlotterNVENC(video_path='in.mp4', roi_path='ROI_definitions.h5', save_path='out.mp4', outline_clr=(0, 0, 255)).run()
    """

    def __init__(self,
                 video_path: Union[str, os.PathLike],
                 geometries: Optional[List[list]] = None,
                 roi_path: Optional[Union[str, os.PathLike]] = None,
                 config_path: Optional[Union[str, os.PathLike]] = None,
                 video_name: Optional[str] = None,
                 save_path: Optional[Union[str, os.PathLike]] = None,
                 colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
                 palette: Optional[str] = None,
                 shape_opacity: float = 0.5,
                 bg_opacity: float = 1.0,
                 outline_clr: Optional[Tuple[int, int, int]] = None,
                 thickness: Optional[int] = 2,
                 circle_size: Optional[int] = None,
                 codec: Literal['h264', 'hevc', 'av1'] = 'h264',
                 buffer_size: int = 16,
                 n_workers: Optional[int] = None,
                 verbose: bool = True):

        n_src = sum(x is not None for x in (geometries, roi_path, config_path))
        if n_src != 1:
            raise InvalidInputError(msg='Pass exactly one geometry source: geometries (list of per-frame shapes), roi_path (ROI .h5), or config_path (SimBA project).', source=self.__class__.__name__)
        self._common_setup(video_path=video_path, save_path=save_path, shape_opacity=shape_opacity, bg_opacity=bg_opacity,
                           outline_clr=outline_clr, thickness=thickness, circle_size=circle_size, codec=codec,
                           buffer_size=buffer_size, n_workers=n_workers, verbose=verbose)
        if geometries is not None:                                  # per-frame shapely geometries
            check_instance(source=f'{self.__class__.__name__} geometries', instance=geometries, accepted_types=(list,))
            if (palette is None) and (colors is None):
                raise InvalidInputError(msg='Pass either colors or palette with geometries.', source=self.__class__.__name__)
            n_geo = len(geometries)
            if n_geo < 1:
                raise InvalidInputError(msg='geometries is empty.', source=self.__class__.__name__)
            # static = geometries don't change frame-to-frame (each track holds a single shape) -> store one frame.
            self._static = all(len(g) == 1 for g in geometries)
            self.colors = self._resolve_colors(colors=colors, palette=palette, n_geo=n_geo)
            self.poly_verts, self.poly_vcount, self.poly_bbox, self.circles = self._prep_geometry_arrays(geometries=geometries, n=self.video_meta['frame_count'], n_geo=n_geo)
        else:                                                        # ROIs from a SimBA .h5 / project (static)
            arrays = self._roi_to_arrays(video_path=video_path, roi_path=roi_path, config_path=config_path, video_name=video_name)
            self._static = True
            self.colors, self.poly_verts, self.poly_vcount, self.poly_bbox, self.circles = (
                arrays['colors'], arrays['poly_verts'], arrays['poly_vcount'], arrays['poly_bbox'], arrays['circles'])

    def _common_setup(self, video_path, save_path, shape_opacity, bg_opacity, outline_clr, thickness,
                      circle_size, codec, buffer_size, n_workers, verbose) -> None:
        """Shared, geometry-independent construction (guards, video meta, output path, worker count, opacities,
        outline). Used by both ``__init__`` and :meth:`from_roi`."""
        get_pkg_version(pkg='PyNvVideoCodec', raise_error=True)
        get_pkg_version(pkg='torch', raise_error=True)
        check_nvidea_gpu_available(raise_error=True)
        check_file_exist_and_readable(file_path=video_path)
        check_float(name=f'{self.__class__.__name__} shape_opacity', value=shape_opacity, min_value=0.0, max_value=1.0)
        check_float(name=f'{self.__class__.__name__} bg_opacity', value=bg_opacity, min_value=0.0, max_value=1.0)
        check_str(name=f'{self.__class__.__name__} codec', value=codec, options=NVENC_CODECS)
        check_int(name=f'{self.__class__.__name__} buffer_size', value=buffer_size, min_value=1)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose')
        if outline_clr is not None:
            check_if_valid_rgb_tuple(data=outline_clr, source=f'{self.__class__.__name__} outline_clr')
            check_int(name=f'{self.__class__.__name__} thickness', value=thickness, min_value=1)
        self.video_meta = get_video_meta_data(video_path=video_path)
        w, h = self.video_meta['width'], self.video_meta['height']
        if circle_size is None:
            circle_size = max(2, int(max(w, h) / 100))
        else:
            check_int(name=f'{self.__class__.__name__} circle_size', value=circle_size, min_value=1)
        self.circle_size = int(circle_size)
        self.outline_clr = np.array(outline_clr if outline_clr is not None else (0, 0, 0), dtype=np.int32)
        self.thickness = int(thickness) if outline_clr is not None else 0
        available = max(1, get_nvdec_count(gpu_name=torch.cuda.get_device_name(0)))
        if n_workers is None:
            n_workers = available
        else:
            check_int(name=f'{self.__class__.__name__} n_workers', value=n_workers, min_value=1)
            if n_workers > available:
                raise SimBAGPUError(msg=f'n_workers={n_workers} requested but GPU has only {available} NVDEC engine(s).', source=self.__class__.__name__)
        video_dir, video_name, _ = get_fn_ext(filepath=video_path)
        if save_path is None:
            save_path = os.path.join(video_dir, f'{video_name}_geometry.mp4')
        else:
            check_if_dir_exists(in_dir=os.path.dirname(save_path), source=f'{self.__class__.__name__} save_path')
        if os.path.abspath(str(save_path)) == os.path.abspath(str(video_path)):
            raise InvalidInputError(msg=f'save_path {save_path} would overwrite the source video.', source=self.__class__.__name__)
        self.video_path, self.save_path = video_path, save_path
        self.shape_opacity, self.bg_opacity = float(shape_opacity), float(bg_opacity)
        self.codec, self.verbose, self.buffer_size, self.n_workers = codec, verbose, buffer_size, n_workers

    @classmethod
    def from_roi(cls,
                 video_path: Union[str, os.PathLike],
                 roi_path: Optional[Union[str, os.PathLike]] = None,
                 config_path: Optional[Union[str, os.PathLike]] = None,
                 video_name: Optional[str] = None,
                 save_path: Optional[Union[str, os.PathLike]] = None,
                 shape_opacity: float = 0.0,
                 bg_opacity: float = 1.0,
                 outline_clr: Optional[Tuple[int, int, int]] = (0, 0, 255),
                 thickness: Optional[int] = 2,
                 codec: Literal['h264', 'hevc', 'av1'] = 'h264',
                 buffer_size: int = 16,
                 n_workers: Optional[int] = None,
                 verbose: bool = True) -> "GeometryPlotterNVENC":
        """Convenience constructor for SimBA ROIs. Identical to passing ``roi_path`` (or ``config_path``) to
        :class:`GeometryPlotterNVENC` directly, but defaults to the usual outline-only ROI look
        (``shape_opacity=0`` with a red outline).

        :example:

        >>> GeometryPlotterNVENC.from_roi(video_path='in.mp4', roi_path='ROI_definitions.h5', save_path='out.mp4').run()
        """
        return cls(video_path=video_path, roi_path=roi_path, config_path=config_path, video_name=video_name,
                   save_path=save_path, shape_opacity=shape_opacity, bg_opacity=bg_opacity, outline_clr=outline_clr,
                   thickness=thickness, codec=codec, buffer_size=buffer_size, n_workers=n_workers, verbose=verbose)

    @classmethod
    def _roi_to_arrays(cls, video_path, roi_path, config_path, video_name) -> dict:
        """Read a SimBA ROI store (standalone ``roi_path`` .h5 or a project ``config_path``) and build the
        static device arrays. Rectangles + polygons become filled polygons; circle ROIs are routed to the
        O(1) circle path (centre + radius) with their own "Color BGR"."""
        if config_path is not None:
            from simba.mixins.config_reader import ConfigReader
            cr = ConfigReader(config_path=config_path, read_video_info=False)
            cr.read_roi_data()
            rect_df, circ_df, poly_df = cr.rectangles_df, cr.circles_df, cr.polygon_df
        else:
            check_file_exist_and_readable(file_path=roi_path)
            rect_df = pd.read_hdf(roi_path, key=Keys.ROI_RECTANGLES.value)
            circ_df = pd.read_hdf(roi_path, key=Keys.ROI_CIRCLES.value).dropna(how='any')
            poly_df = pd.read_hdf(roi_path, key=Keys.ROI_POLYGONS.value).dropna(how='any')
        vname = video_name if video_name is not None else get_fn_ext(filepath=video_path)[1]
        # rectangles + polygons -> filled polygons; circles -> O(1) circle path (own radius, not buffered polygon)
        geo_dict, clr_dict = GeometryMixin.simba_roi_to_geometries(rectangles_df=rect_df, polygons_df=poly_df, color=True)
        poly_specs = []
        if vname in geo_dict:
            for nm, geom in geo_dict[vname].items():
                v = cls._shape_to_polygon_coords(geom)
                if v is not None:
                    poly_specs.append((v, tuple(int(c) for c in clr_dict[vname][nm])))
        circ_specs = []
        if (circ_df is not None) and (len(circ_df) > 0):
            for _, row in circ_df[circ_df['Video'] == vname].iterrows():
                center = row['Tags']['Center tag']
                circ_specs.append((int(center[0]), int(center[1]), int(row['radius']), tuple(int(c) for c in row['Color BGR'])))
        n_geo = len(poly_specs) + len(circ_specs)
        if n_geo == 0:
            raise InvalidInputError(msg=f'No ROIs found for video "{vname}".', source=cls.__name__)
        max_v = max([3] + [v.shape[0] for v, _ in poly_specs])
        poly_verts = np.zeros((1, n_geo, max_v, 2), dtype=np.int32)
        poly_vcount = np.zeros((1, n_geo), dtype=np.int32)
        poly_bbox = np.zeros((1, n_geo, 4), dtype=np.int32)
        circles = np.zeros((1, n_geo, 3), dtype=np.int32)
        colors = np.zeros((n_geo, 3), dtype=np.int32)
        for i, (v, clr) in enumerate(poly_specs):
            k = v.shape[0]
            poly_verts[0, i, :k] = v
            poly_vcount[0, i] = k
            poly_bbox[0, i] = (v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max())
            colors[i] = clr
        for j, (cx, cy, r, clr) in enumerate(circ_specs):
            gi = len(poly_specs) + j
            circles[0, gi] = (cx, cy, r)
            colors[gi] = clr
        return {'colors': colors, 'poly_verts': poly_verts, 'poly_vcount': poly_vcount, 'poly_bbox': poly_bbox, 'circles': circles}

    @staticmethod
    def _resolve_colors(colors, palette, n_geo) -> np.ndarray:
        """Return an (n_geo, 3) int32 BGR colour array from a colours list or a palette name."""
        if palette is not None:
            pal = create_color_palettes(no_animals=1, map_size=n_geo + 1, cmaps=[palette])
            flat = [x for xs in pal for x in xs]
            return np.array(flat[:n_geo], dtype=np.int32)
        color_dict = get_color_dict()
        check_valid_lst(data=colors, source='GeometryPlotterNVENC colors', valid_dtypes=(str, tuple), exact_len=n_geo)
        out = np.zeros((n_geo, 3), dtype=np.int32)
        for i, clr in enumerate(colors):
            out[i] = np.array(color_dict[clr] if isinstance(clr, str) else clr, dtype=np.int32)
        return out

    @staticmethod
    def _shape_to_circle(shape, radius: int):
        """Return (x, y, r) if ``shape`` is a plottable Point, else None."""
        if isinstance(shape, Point):
            arr = np.array(shape.coords)
            if arr.size >= 2 and np.isfinite(arr).all():
                return int(arr.flat[0]), int(arr.flat[1]), radius
        return None

    @staticmethod
    def _shape_to_polygon_coords(shape):
        """Return an (N, 2) int32 vertex array for a fillable shape, else None. Multi* / LineString
        are approximated by their exterior / convex-hull polygon."""
        if isinstance(shape, Polygon):
            coords = np.array(shape.exterior.coords)
        elif isinstance(shape, LineString):
            coords = np.array(shape.coords)
        elif isinstance(shape, MultiPolygon):
            coords = np.array(shape.convex_hull.exterior.coords)
        elif isinstance(shape, MultiLineString):
            coords = np.array(shape.convex_hull.exterior.coords)
        else:
            return None
        if coords.ndim != 2 or coords.shape[0] < 3 or not np.isfinite(coords).all():
            return None
        return coords.astype(np.int32)

    def _prep_geometry_arrays(self, geometries, n, n_geo):
        """Convert per-frame shapely geometries into padded device-ready int32 arrays. For static
        geometries (e.g. ROIs) only a single frame is stored and reused for every output frame."""
        n_eff = 1 if self._static else n
        max_v = 3
        for g in geometries:
            for shape in g:
                v = self._shape_to_polygon_coords(shape)
                if v is not None:
                    max_v = max(max_v, v.shape[0])
        poly_verts = np.zeros((n_eff, n_geo, max_v, 2), dtype=np.int32)
        poly_vcount = np.zeros((n_eff, n_geo), dtype=np.int32)
        poly_bbox = np.zeros((n_eff, n_geo, 4), dtype=np.int32)
        circles = np.zeros((n_eff, n_geo, 3), dtype=np.int32)
        for gi, g in enumerate(geometries):
            for fi in range(min(n_eff, len(g))):
                shape = g[fi]
                circ = self._shape_to_circle(shape, self.circle_size)
                if circ is not None:
                    circles[fi, gi] = circ
                    continue
                v = self._shape_to_polygon_coords(shape)
                if v is not None:
                    k = min(v.shape[0], max_v)
                    poly_verts[fi, gi, :k] = v[:k]
                    poly_vcount[fi, gi] = k
                    poly_bbox[fi, gi] = (v[:k, 0].min(), v[:k, 1].min(), v[:k, 0].max(), v[:k, 1].max())
        return poly_verts, poly_vcount, poly_bbox, circles

    def _encode_chunk(self, start: int, end: int, out_path: str) -> None:
        """Decode -> geometry-overlay kernel -> NVENC-encode frames [start, end) and mux to ``out_path``."""
        w, h, n = self.video_meta['width'], self.video_meta['height'], self.video_meta['frame_count']
        argb_t = torch.empty((1, h, w, 4), dtype=torch.uint8, device='cuda:0')
        argb_dev = cuda.as_cuda_array(argb_t)[0]
        pv_dev = cuda.to_device(self.poly_verts)
        pc_dev = cuda.to_device(self.poly_vcount)
        bb_dev = cuda.to_device(self.poly_bbox)
        ci_dev = cuda.to_device(self.circles)
        clr_dev = cuda.to_device(self.colors)
        outline_dev = cuda.to_device(self.outline_clr)
        decoder = get_nvc_decoder(video_path=self.video_path, use_device_memory=True, threaded=True, buffer_size=self.buffer_size, start_frame=start)
        encoder = get_nvc_encoder(width=w, height=h, codec=self.codec, fmt='ARGB')
        grid = (math.ceil(w / TPB[0]), math.ceil(h / TPB[1]))
        raw_path = f'{out_path}.raw.{self.codec}'
        i = start
        with open(raw_path, 'wb') as raw:
            while i < end:
                frames = decoder.get_batch_frames(min(self.buffer_size, end - i))
                if not frames:
                    break
                for f in frames:
                    if i >= end:
                        break
                    rgb = cuda.as_cuda_array(torch.from_dlpack(f))
                    fi = 0 if self._static else i
                    _geometry_overlay_kernel[grid, TPB](rgb, argb_dev, pv_dev[fi], pc_dev[fi], bb_dev[fi], ci_dev[fi], clr_dev, self.shape_opacity, self.bg_opacity, outline_dev, self.thickness)
                    bitstream = encoder.Encode(argb_t[0])
                    if bitstream:
                        raw.write(bytes(bitstream))
                    i += 1
                    if self.verbose and i % 500 == 0:
                        stdout_information(msg=f'Encoded {i}/{n} frames...', source=self.__class__.__name__)
            tail = encoder.EndEncode()
            if tail:
                raw.write(bytes(tail))
        decoder.end()
        subprocess.call(f'ffmpeg -y -loglevel error -framerate {self.video_meta["fps"]} -i "{raw_path}" -c copy "{out_path}"', shell=True)
        try:
            os.remove(raw_path)
        except OSError:
            pass

    def run(self) -> None:
        """Render the geometry-overlay video on the GPU and write it to ``save_path``."""
        timer = SimbaTimer(start=True)
        n = self.video_meta['frame_count']
        if self.n_workers <= 1:
            self._encode_chunk(0, n, str(self.save_path))
        else:
            bounds = np.linspace(0, n, self.n_workers + 1).astype(int)
            tmp = tempfile.gettempdir()
            chunk_paths = [os.path.join(tmp, f'_geochunk_{os.getpid()}_{k}.mp4') for k in range(self.n_workers)]
            errors, threads = [], []

            def _work(k: int) -> None:
                try:
                    self._encode_chunk(int(bounds[k]), int(bounds[k + 1]), chunk_paths[k])
                except Exception as e:
                    errors.append((k, e))

            for k in range(self.n_workers):
                t = threading.Thread(target=_work, args=(k,), daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            if errors:
                k, exc = errors[0]
                raise FrameRangeError(msg=f'{self.__class__.__name__} chunk {k} failed: {exc}', source=self.__class__.__name__)
            list_path = os.path.join(tmp, f'_geolist_{os.getpid()}.txt')
            with open(list_path, 'w') as fh:
                for p in chunk_paths:
                    fh.write(f"file '{p}'\n")
            subprocess.call(f'ffmpeg -y -loglevel error -f concat -safe 0 -i "{list_path}" -c copy "{self.save_path}"', shell=True)
            for p in chunk_paths + [list_path]:
                try:
                    os.remove(p)
                except OSError:
                    pass
        timer.stop_timer()
        if self.verbose:
            stdout_success(msg=f'Geometry overlay video saved at {self.save_path}.', elapsed_time=timer.elapsed_time_str, source=self.__class__.__name__)


if __name__ == '__main__':
    GeometryPlotterNVENC(geometries=None,
                         roi_path=r"D:\troubleshooting\open_field_below\project_folder\logs\measures\ROI_definitions.h5",

                         video_path=r"D:\troubleshooting\open_field_below\project_folder\videos\raw_clip1.mp4", save_path=r'D:\troubleshooting\open_field_below\project_folder\videos\.mp4',
                         colors=['Red']).run()
    pass
