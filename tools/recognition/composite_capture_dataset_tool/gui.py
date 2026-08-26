from __future__ import annotations

import random
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from PIL import Image, ImageDraw, ImageTk

from .coco import CocoDataset
from .image_io import load_coco_image
from .composer import CompositeResult, compose_capture_image
from .dataset import OutputDatasetManager
from .geometry import constrain_drag_rect
from .layout import REGION_SPECS
from .models import Rect, RegionSelection


REGION_COLORS = {
    "completed_hand": "#00d084",
    "dora_indicators": "#ffd166",
    "melds": "#4dabf7",
}

HAND_DORA_RANDOM_CROPS = (
    Rect(141, 835, 323, 76),
    Rect(489, 830, 323, 76),
    Rect(10, 412, 391, 92),
    Rect(557, 457, 391, 92),
    Rect(161, 50, 323, 76),
    Rect(477, 49, 357, 84),
)

MELDS_RANDOM_CROPS = (
    Rect(601, 398, 191, 191),
    Rect(323, 603, 235, 235),
    Rect(170, 326, 222, 222),
    Rect(424, 151, 211, 211),
)


class CompositeCaptureApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        dataset: CocoDataset,
        output_manager: OutputDatasetManager,
        annotation_selection_policy: str,
        min_retained_area_ratio: float,
        start_index: int = 0,
    ) -> None:
        self.root = root
        self.dataset = dataset
        self.output_manager = output_manager
        self.annotation_selection_policy = annotation_selection_policy
        self.min_retained_area_ratio = min_retained_area_ratio
        self.current_index = min(max(start_index, 0), len(dataset.images) - 1)
        self.current_image_record: dict[str, Any] | None = None
        self.source_image: Image.Image | None = None
        self.source_annotations: list[dict[str, Any]] = []
        self.selections: dict[str, RegionSelection] = {}
        self._drag_anchor: tuple[int, int] | None = None
        self._display_scale = 1.0
        self._display_offset = (0, 0)
        self._display_size = (1, 1)
        self._canvas_photo: ImageTk.PhotoImage | None = None

        self.active_region = tk.StringVar(value=next(iter(REGION_SPECS)))
        self.enabled_vars = {
            key: tk.BooleanVar(value=False) for key in REGION_SPECS
        }
        self.rotation_vars = {
            key: tk.StringVar(value="0") for key in REGION_SPECS
        }
        self.selection_text_vars = {
            key: tk.StringVar(value="not selected") for key in REGION_SPECS
        }
        self.show_source_annotations = tk.BooleanVar(value=True)
        self.index_var = tk.StringVar(value=str(self.current_index + 1))
        self.image_info_var = tk.StringVar()
        self.output_info_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Loading image...")

        self._build_ui()
        self._load_current_image()

    def _build_ui(self) -> None:
        self.root.title("Composite capture COCO dataset tool")
        self.root.geometry("1420x900")
        self.root.minsize(1050, 700)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)

        canvas_frame = ttk.Frame(self.root, padding=(8, 8, 4, 4))
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_frame,
            background="#202020",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)

        controls = ttk.Frame(self.root, padding=(8, 10, 12, 8), width=390)
        controls.grid(row=0, column=1, sticky="ns")
        controls.grid_propagate(False)
        controls.columnconfigure(0, weight=1)

        navigation = ttk.LabelFrame(controls, text="Source image", padding=8)
        navigation.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        navigation.columnconfigure(1, weight=1)
        ttk.Button(navigation, text="Previous", command=self._previous_image).grid(
            row=0, column=0, padx=(0, 4)
        )
        index_entry = ttk.Entry(navigation, textvariable=self.index_var, width=8)
        index_entry.grid(row=0, column=1, sticky="ew", padx=4)
        index_entry.bind("<Return>", lambda _event: self._jump_to_image())
        ttk.Button(navigation, text="Go", command=self._jump_to_image).grid(
            row=0, column=2, padx=4
        )
        ttk.Button(navigation, text="Next", command=self._next_image).grid(
            row=0, column=3, padx=(4, 0)
        )
        ttk.Label(
            navigation,
            textvariable=self.image_info_var,
            wraplength=345,
            justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            navigation,
            text="Show source tile annotations",
            variable=self.show_source_annotations,
            command=self._draw_overlays,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))

        regions_frame = ttk.LabelFrame(
            controls,
            text="Capture regions",
            padding=8,
        )
        regions_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        regions_frame.columnconfigure(2, weight=1)
        ttk.Label(
            regions_frame,
            text=(
                "Choose an active region, then drag on the image. "
                "The crop is constrained to the production aspect ratio."
            ),
            wraplength=345,
            justify="left",
        ).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 8))

        row = 1
        for region_key, spec in REGION_SPECS.items():
            ttk.Radiobutton(
                regions_frame,
                variable=self.active_region,
                value=region_key,
            ).grid(row=row, column=0, sticky="w")
            ttk.Checkbutton(
                regions_frame,
                text=spec.label,
                variable=self.enabled_vars[region_key],
                command=lambda key=region_key: self._toggle_region(key),
            ).grid(row=row, column=1, sticky="w", padx=(2, 8))
            rotation = ttk.Combobox(
                regions_frame,
                textvariable=self.rotation_vars[region_key],
                values=("0", "90", "180", "270"),
                width=5,
                state="readonly",
            )
            rotation.grid(row=row, column=2, sticky="e")
            rotation.bind(
                "<<ComboboxSelected>>",
                lambda _event, key=region_key: self._rotation_changed(key),
            )
            ttk.Label(regions_frame, text="° CW").grid(
                row=row, column=3, sticky="w", padx=(2, 6)
            )
            ttk.Button(
                regions_frame,
                text="Clear",
                width=7,
                command=lambda key=region_key: self._clear_region(key),
            ).grid(row=row, column=4, sticky="e")
            ttk.Label(
                regions_frame,
                textvariable=self.selection_text_vars[region_key],
                wraplength=330,
                justify="left",
            ).grid(
                row=row + 1,
                column=1,
                columnspan=4,
                sticky="w",
                pady=(1, 7),
            )
            row += 2

        ttk.Button(
            regions_frame,
            text="Random preset crops",
            command=self._set_random_preset_crops,
        ).grid(row=row, column=0, columnspan=5, sticky="ew", pady=(3, 0))
        row += 1
        ttk.Button(
            regions_frame,
            text="Clear all regions",
            command=self._clear_all_regions,
        ).grid(row=row, column=0, columnspan=5, sticky="ew", pady=(4, 0))

        output_frame = ttk.LabelFrame(controls, text="Composite output", padding=8)
        output_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        output_frame.columnconfigure(0, weight=1)
        ttk.Label(
            output_frame,
            text=(
                "Disabled regions remain black. Source annotations are selected "
                f"with the '{self.annotation_selection_policy}' policy and clipped "
                "to the chosen crop before transformation. A bbox is retained only "
                f"when more than {self.min_retained_area_ratio:.0%} of its original "
                "area remains inside the crop."
            ),
            wraplength=345,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(
            output_frame,
            text="Preview composite",
            command=self._preview_composite,
        ).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(10, 0))
        ttk.Button(
            output_frame,
            text="Save composite",
            command=self._save_composite,
        ).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(10, 0))
        ttk.Label(
            output_frame,
            textvariable=self.output_info_var,
            wraplength=345,
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(
            controls,
            textvariable=self.status_var,
            wraplength=360,
            justify="left",
            relief="sunken",
            padding=7,
        ).grid(row=3, column=0, sticky="sew")
        controls.rowconfigure(3, weight=1)

        self.root.bind("<Left>", lambda _event: self._previous_image())
        self.root.bind("<Right>", lambda _event: self._next_image())

    def _load_current_image(self) -> None:
        try:
            record = self.dataset.image_at(self.current_index)
            image_path = self.dataset.resolve_image_path(record)
            loaded_image = load_coco_image(image_path)
            image = loaded_image.image
        except Exception as error:
            messagebox.showerror("Unable to load source image", str(error))
            self.status_var.set(str(error))
            return

        self.current_image_record = record
        self.source_image = image
        self.source_annotations = self.dataset.annotations_for_image(int(record["id"]))
        self.selections.clear()
        for region_key in REGION_SPECS:
            self.enabled_vars[region_key].set(False)
            self.selection_text_vars[region_key].set("not selected")
        self._drag_anchor = None
        self.index_var.set(str(self.current_index + 1))

        declared_size = (int(record["width"]), int(record["height"]))
        size_warning = ""
        if declared_size != image.size:
            size_warning = (
                f"\nWARNING: COCO size {declared_size} differs from EXIF-oriented "
                f"file size {image.size}."
            )
        orientation_info = ""
        if loaded_image.exif_transpose_applied:
            orientation_info = (
                f"\nEXIF orientation={loaded_image.exif_orientation}; "
                f"raw={loaded_image.raw_size[0]}×{loaded_image.raw_size[1]} → "
                f"oriented={image.size[0]}×{image.size[1]}"
            )
        self.image_info_var.set(
            f"{self.current_index + 1} / {len(self.dataset.images)}\n"
            f"id={record['id']}  annotations={len(self.source_annotations)}\n"
            f"{record['file_name']}\nsize={image.size[0]}×{image.size[1]}"
            f"{orientation_info}{size_warning}"
        )
        self._update_output_info()
        self.status_var.set(
            f"Loaded {record['file_name']}. Select a region and drag on the image."
        )
        self.root.after_idle(self._render_canvas)

    def _on_canvas_configure(self, _event: tk.Event[Any]) -> None:
        if self.source_image is not None:
            self._render_canvas()

    def _render_canvas(self) -> None:
        if self.source_image is None:
            return
        canvas_width = max(self.canvas.winfo_width(), 50)
        canvas_height = max(self.canvas.winfo_height(), 50)
        source_width, source_height = self.source_image.size
        scale = min(canvas_width / source_width, canvas_height / source_height)
        display_width = max(1, int(round(source_width * scale)))
        display_height = max(1, int(round(source_height * scale)))
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2

        display_image = self.source_image.resize(
            (display_width, display_height),
            resample=Image.Resampling.LANCZOS,
        )
        self._canvas_photo = ImageTk.PhotoImage(display_image)
        self.canvas.delete("all")
        self.canvas.create_image(
            offset_x,
            offset_y,
            image=self._canvas_photo,
            anchor="nw",
            tags=("source_image",),
        )
        self._display_scale = scale
        self._display_offset = (offset_x, offset_y)
        self._display_size = (display_width, display_height)
        self._draw_overlays()

    def _draw_overlays(self) -> None:
        if self.source_image is None:
            return
        self.canvas.delete("source_bbox")
        self.canvas.delete("selection")
        if self.show_source_annotations.get():
            image_bounds = Rect(0, 0, *self.source_image.size)
            for annotation in self.source_annotations:
                try:
                    bbox = Rect.from_coco_bbox(
                        annotation.get("bbox"),
                        context=f"annotation {annotation.get('id')}",
                    ).intersection(image_bounds)
                except ValueError:
                    continue
                if bbox is None:
                    continue
                left, top = self._source_to_canvas(bbox.x, bbox.y)
                right, bottom = self._source_to_canvas(bbox.right, bbox.bottom)
                self.canvas.create_rectangle(
                    left,
                    top,
                    right,
                    bottom,
                    outline="#f4f4f4",
                    width=1,
                    tags=("source_bbox",),
                )

        for region_key, selection in self.selections.items():
            color = REGION_COLORS[region_key]
            left, top = self._source_to_canvas(selection.crop.x, selection.crop.y)
            right, bottom = self._source_to_canvas(
                selection.crop.right,
                selection.crop.bottom,
            )
            enabled = self.enabled_vars[region_key].get()
            self.canvas.create_rectangle(
                left,
                top,
                right,
                bottom,
                outline=color,
                width=3 if enabled else 2,
                dash=() if enabled else (6, 4),
                tags=("selection",),
            )
            self.canvas.create_text(
                left + 5,
                top + 5,
                text=(
                    f"{REGION_SPECS[region_key].label}  "
                    f"{selection.rotation_clockwise}°"
                ),
                fill=color,
                anchor="nw",
                font=("TkDefaultFont", 10, "bold"),
                tags=("selection",),
            )

    def _source_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return (
            self._display_offset[0] + x * self._display_scale,
            self._display_offset[1] + y * self._display_scale,
        )

    def _canvas_to_source(self, x: int, y: int) -> tuple[int, int] | None:
        if self.source_image is None:
            return None
        offset_x, offset_y = self._display_offset
        display_width, display_height = self._display_size
        if not (
            offset_x <= x <= offset_x + display_width
            and offset_y <= y <= offset_y + display_height
        ):
            return None
        source_x = int(round((x - offset_x) / self._display_scale))
        source_y = int(round((y - offset_y) / self._display_scale))
        source_x = min(max(source_x, 0), self.source_image.size[0])
        source_y = min(max(source_y, 0), self.source_image.size[1])
        return (source_x, source_y)

    def _on_mouse_press(self, event: tk.Event[Any]) -> None:
        source_point = self._canvas_to_source(event.x, event.y)
        if source_point is None:
            return
        self._drag_anchor = source_point
        region_key = self.active_region.get()
        self.enabled_vars[region_key].set(True)

    def _on_mouse_drag(self, event: tk.Event[Any]) -> None:
        if self._drag_anchor is None or self.source_image is None:
            return
        source_point = self._canvas_to_source(event.x, event.y)
        if source_point is None:
            return
        region_key = self.active_region.get()
        rotation = int(self.rotation_vars[region_key].get())
        aspect = REGION_SPECS[region_key].source_aspect_for_rotation(rotation)
        crop = constrain_drag_rect(
            self._drag_anchor,
            source_point,
            self.source_image.size,
            aspect,
        )
        if crop is None:
            return
        self.selections[region_key] = RegionSelection(
            region_key=region_key,
            crop=crop,
            rotation_clockwise=rotation,
        )
        self.enabled_vars[region_key].set(True)
        self._update_selection_text(region_key)
        self._draw_overlays()

    def _on_mouse_release(self, event: tk.Event[Any]) -> None:
        if self._drag_anchor is None:
            return
        self._on_mouse_drag(event)
        self._drag_anchor = None
        region_key = self.active_region.get()
        if region_key in self.selections:
            self.status_var.set(
                f"Selected {REGION_SPECS[region_key].label}. "
                "Preview or save, or select another region."
            )

    def _toggle_region(self, region_key: str) -> None:
        if self.enabled_vars[region_key].get() and region_key not in self.selections:
            self.active_region.set(region_key)
            self.status_var.set(
                f"Drag a {REGION_SPECS[region_key].label} crop on the source image."
            )
        self._draw_overlays()

    def _rotation_changed(self, region_key: str) -> None:
        if region_key in self.selections:
            del self.selections[region_key]
            self.selection_text_vars[region_key].set("not selected; redraw required")
        self.active_region.set(region_key)
        self.status_var.set(
            f"Rotation changed to {self.rotation_vars[region_key].get()}° clockwise. "
            "Redraw this region because its source aspect may have changed."
        )
        self._draw_overlays()

    def _clear_region(self, region_key: str) -> None:
        self.selections.pop(region_key, None)
        self.enabled_vars[region_key].set(False)
        self.selection_text_vars[region_key].set("not selected")
        self._draw_overlays()

    def _set_random_preset_crops(self) -> None:
        if self.source_image is None:
            return

        image_bounds = Rect(0, 0, *self.source_image.size)
        hand_dora_candidates = [
            crop
            for crop in HAND_DORA_RANDOM_CROPS
            if image_bounds.contains_rect(crop)
        ]
        meld_candidates = [
            crop for crop in MELDS_RANDOM_CROPS if image_bounds.contains_rect(crop)
        ]
        if len(hand_dora_candidates) < 2 or not meld_candidates:
            messagebox.showerror(
                "Random preset crops unavailable",
                "The current source image is too small for the configured preset crops: "
                f"image={self.source_image.size}",
            )
            return

        completed_crop, dora_crop = random.sample(hand_dora_candidates, 2)
        selected_crops = {
            "completed_hand": completed_crop,
            "dora_indicators": dora_crop,
            "melds": random.choice(meld_candidates),
        }
        for region_key, crop in selected_crops.items():
            self.selections[region_key] = RegionSelection(
                region_key=region_key,
                crop=crop,
                rotation_clockwise=0,
            )
            self.enabled_vars[region_key].set(True)
            self.rotation_vars[region_key].set("0")
            self._update_selection_text(region_key)

        self.active_region.set("completed_hand")
        self._draw_overlays()
        self.status_var.set(
            "Random preset crops selected for completed hand, dora indicators, "
            "and melds. Preview or save the composite."
        )

    def _clear_all_regions(self) -> None:
        self.selections.clear()
        for region_key in REGION_SPECS:
            self.enabled_vars[region_key].set(False)
            self.selection_text_vars[region_key].set("not selected")
        self._draw_overlays()
        self.status_var.set("All capture regions were cleared.")

    def _update_selection_text(self, region_key: str) -> None:
        selection = self.selections[region_key]
        crop = selection.crop
        self.selection_text_vars[region_key].set(
            f"crop=({int(crop.x)}, {int(crop.y)}, "
            f"{int(crop.width)}, {int(crop.height)})  "
            f"rotation={selection.rotation_clockwise}°"
        )

    def _effective_selections(self) -> dict[str, RegionSelection]:
        enabled_without_selection = [
            REGION_SPECS[key].label
            for key in REGION_SPECS
            if self.enabled_vars[key].get() and key not in self.selections
        ]
        if enabled_without_selection:
            raise ValueError(
                "Enabled regions still need a crop: "
                + ", ".join(enabled_without_selection)
            )
        selections = {
            key: self.selections[key]
            for key in REGION_SPECS
            if self.enabled_vars[key].get() and key in self.selections
        }
        if not selections:
            raise ValueError("Select and enable at least one capture region")
        return selections

    def _compose_current(self) -> tuple[CompositeResult, dict[str, RegionSelection]]:
        if self.source_image is None or self.current_image_record is None:
            raise RuntimeError("No source image is loaded")
        selections = self._effective_selections()
        result = compose_capture_image(
            self.source_image,
            selections,
            self.source_annotations,
            annotation_selection_policy=self.annotation_selection_policy,
            min_retained_area_ratio=self.min_retained_area_ratio,
        )
        return result, selections

    def _preview_composite(self) -> None:
        try:
            result, _selections = self._compose_current()
        except Exception as error:
            messagebox.showerror("Unable to preview composite", str(error))
            return

        preview = result.image.copy()
        draw = ImageDraw.Draw(preview)
        for annotation in result.annotations:
            color = REGION_COLORS[annotation.region_key]
            draw.rectangle(
                (
                    annotation.bbox.x,
                    annotation.bbox.y,
                    annotation.bbox.right,
                    annotation.bbox.bottom,
                ),
                outline=color,
                width=1,
            )
        preview = preview.resize((640, 640), Image.Resampling.NEAREST)
        window = tk.Toplevel(self.root)
        window.title(
            f"Composite preview — {len(result.annotations)} tile annotations"
        )
        photo = ImageTk.PhotoImage(preview)
        label = ttk.Label(window, image=photo)
        label.image = photo  # type: ignore[attr-defined]
        label.pack(padx=10, pady=10)
        stats = ", ".join(
            f"{REGION_SPECS[key].label}: {value.retained_annotations}"
            + (
                f" ({value.clipped_annotations} clipped)"
                if value.clipped_annotations
                else ""
            )
            for key, value in result.stats_by_region.items()
        )
        ttk.Label(window, text=stats, padding=(10, 0, 10, 10)).pack()

    def _save_composite(self) -> None:
        if self.current_image_record is None:
            return
        try:
            result, selections = self._compose_current()
            saved = self.output_manager.save_composite(
                result,
                source_annotation_path=self.dataset.annotation_path,
                source_image=self.current_image_record,
                selections=selections,
                annotation_selection_policy=self.annotation_selection_policy,
                min_retained_area_ratio=self.min_retained_area_ratio,
            )
        except Exception as error:
            messagebox.showerror("Unable to save composite", str(error))
            self.status_var.set(str(error))
            return

        self._update_output_info()
        self.status_var.set(
            f"Saved image id {saved.image_id} with {saved.annotation_count} "
            f"annotations: {Path(saved.image_path).name}. "
            "Selections remain active so another composite can be saved from "
            "the same source image."
        )

    def _update_output_info(self) -> None:
        self.output_info_var.set(
            f"Output: {self.output_manager.output_directory}\n"
            f"images={self.output_manager.image_count}, "
            f"annotations={self.output_manager.annotation_count}"
        )

    def _previous_image(self) -> None:
        if self.current_index <= 0:
            return
        self.current_index -= 1
        self._load_current_image()

    def _next_image(self) -> None:
        if self.current_index >= len(self.dataset.images) - 1:
            return
        self.current_index += 1
        self._load_current_image()

    def _jump_to_image(self) -> None:
        try:
            one_based_index = int(self.index_var.get())
        except ValueError:
            messagebox.showerror("Invalid image index", self.index_var.get())
            return
        if not 1 <= one_based_index <= len(self.dataset.images):
            messagebox.showerror(
                "Invalid image index",
                f"Enter a value from 1 to {len(self.dataset.images)}.",
            )
            return
        self.current_index = one_based_index - 1
        self._load_current_image()
