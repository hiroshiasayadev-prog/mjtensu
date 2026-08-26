import {
  containsPoint,
  corners,
  distance,
  normalizeAngle,
  rotationHandle,
  splitBox,
} from './geometry';
import type { Point } from './geometry';
import type { AnnotationBox } from './types';

type DragState =
  | { kind: 'move'; boxId: string; start: Point; original: AnnotationBox }
  | {
      kind: 'resize';
      boxId: string;
      original: AnnotationBox;
      anchor: Point;
      signX: -1 | 1;
      signY: -1 | 1;
    }
  | { kind: 'rotate'; boxId: string; original: AnnotationBox }
  | { kind: 'create'; boxId: string; start: Point };

export class CanvasEditor {
  private readonly context: CanvasRenderingContext2D;
  private image: HTMLImageElement | null = null;
  private boxes: AnnotationBox[] = [];
  private labels = new Map<string, { text: string; tentative: boolean }>();
  private selectedId: string | null = null;
  private drag: DragState | null = null;
  private addMode = false;
  private imageGeneration = 0;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly onChange: (boxes: AnnotationBox[]) => void,
    private readonly onSelection: (box: AnnotationBox | null) => void,
  ) {
    const context = canvas.getContext('2d');
    if (context === null) throw new Error('Canvas 2D context is unavailable.');
    this.context = context;
    canvas.tabIndex = 0;
    canvas.addEventListener('pointerdown', this.onPointerDown);
    canvas.addEventListener('pointermove', this.onPointerMove);
    canvas.addEventListener('pointerup', this.onPointerUp);
    canvas.addEventListener('pointercancel', this.onPointerUp);
    canvas.addEventListener('keydown', (event) => {
      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault();
        this.deleteSelected();
      }
    });
  }

  async setImage(url: string, expectedWidth: number, expectedHeight: number): Promise<boolean> {
    const generation = ++this.imageGeneration;
    const image = new Image();
    image.decoding = 'async';
    image.src = url;
    await image.decode();
    if (generation !== this.imageGeneration) return false;
    if (image.naturalWidth !== expectedWidth || image.naturalHeight !== expectedHeight) {
      throw new Error(
        `Region image size ${image.naturalWidth}x${image.naturalHeight} `
        + `does not match manifest ${expectedWidth}x${expectedHeight}.`,
      );
    }
    this.image = image;
    this.canvas.width = expectedWidth;
    this.canvas.height = expectedHeight;
    this.draw();
    return true;
  }

  clearImage(): void {
    this.imageGeneration += 1;
    this.image = null;
    this.canvas.width = 1;
    this.canvas.height = 1;
    this.context.clearRect(0, 0, 1, 1);
  }

  setBoxes(boxes: AnnotationBox[]): void {
    this.boxes = boxes.map((box) => ({ ...box }));
    if (this.selectedId !== null && !this.boxes.some((box) => box.id === this.selectedId)) {
      this.selectedId = null;
      this.onSelection(null);
    }
    this.draw();
  }

  setLabels(labels: Map<string, { text: string; tentative: boolean }>): void {
    this.labels = labels;
    this.draw();
  }

  setAddMode(enabled: boolean): void {
    this.addMode = enabled;
    this.canvas.classList.toggle('add-mode', enabled);
  }

  isAddMode(): boolean {
    return this.addMode;
  }

  selectedBox(): AnnotationBox | null {
    return this.selectedId === null
      ? null
      : this.boxes.find((box) => box.id === this.selectedId) ?? null;
  }

  deleteSelected(): void {
    if (this.selectedId === null) return;
    const next = this.boxes.filter((box) => box.id !== this.selectedId);
    this.selectedId = null;
    this.commit(next);
    this.onSelection(null);
  }

  splitSelected(direction: 'screen-x' | 'screen-y'): void {
    const selected = this.selectedBox();
    if (selected === null) return;
    const angle = selected.angleDeg * Math.PI / 180;
    const localXAxisIsMoreHorizontal = Math.abs(Math.cos(angle)) >= Math.abs(Math.sin(angle));
    const axis: 'x' | 'y' = direction === 'screen-x'
      ? (localXAxisIsMoreHorizontal ? 'x' : 'y')
      : (localXAxisIsMoreHorizontal ? 'y' : 'x');
    const [first, second] = splitBox(selected, axis, Math.max(1, 2 / this.displayScale()));
    const index = this.boxes.findIndex((box) => box.id === selected.id);
    const next = [...this.boxes];
    next.splice(index, 1, first, second);
    this.selectedId = first.id;
    this.commit(next);
    this.onSelection(first);
  }

  setSelectedAngle(angleDeg: number): void {
    const selected = this.selectedBox();
    if (selected === null || !Number.isFinite(angleDeg)) return;
    const next = this.boxes.map((box) => (
      box.id === selected.id ? { ...box, angleDeg: normalizeAngle(angleDeg) } : box
    ));
    this.commit(next);
    this.onSelection(next.find((box) => box.id === selected.id) ?? null);
  }

  private readonly onPointerDown = (event: PointerEvent): void => {
    if (this.image === null) return;
    this.canvas.focus();
    const point = this.eventPoint(event);
    const handleRadius = 9 / this.displayScale();
    const rotationOffset = 28 / this.displayScale();
    const selected = this.selectedBox();

    if (selected !== null) {
      if (distance(point, rotationHandle(selected, rotationOffset)) <= handleRadius * 1.4) {
        this.drag = { kind: 'rotate', boxId: selected.id, original: { ...selected } };
        this.canvas.setPointerCapture(event.pointerId);
        return;
      }
      const selectedCorners = corners(selected);
      const cornerIndex = selectedCorners.findIndex(
        (corner) => distance(point, corner) <= handleRadius,
      );
      if (cornerIndex >= 0) {
        const anchor = selectedCorners[(cornerIndex + 2) % 4];
        if (anchor === undefined) return;
        const signs = [
          { x: -1 as const, y: -1 as const },
          { x: 1 as const, y: -1 as const },
          { x: 1 as const, y: 1 as const },
          { x: -1 as const, y: 1 as const },
        ];
        const sign = signs[cornerIndex];
        if (sign === undefined) return;
        this.drag = {
          kind: 'resize',
          boxId: selected.id,
          original: { ...selected },
          anchor,
          signX: sign.x,
          signY: sign.y,
        };
        this.canvas.setPointerCapture(event.pointerId);
        return;
      }
    }

    for (let index = this.boxes.length - 1; index >= 0; index -= 1) {
      const candidate = this.boxes[index];
      if (candidate !== undefined && containsPoint(candidate, point)) {
        this.selectedId = candidate.id;
        this.drag = {
          kind: 'move',
          boxId: candidate.id,
          start: point,
          original: { ...candidate },
        };
        this.onSelection(candidate);
        this.canvas.setPointerCapture(event.pointerId);
        this.draw();
        return;
      }
    }

    if (this.addMode) {
      const box: AnnotationBox = {
        id: crypto.randomUUID(),
        centerX: point.x,
        centerY: point.y,
        width: 2,
        height: 2,
        angleDeg: 0,
      };
      this.selectedId = box.id;
      this.drag = { kind: 'create', boxId: box.id, start: point };
      this.boxes = [...this.boxes, box];
      this.onSelection(box);
      this.canvas.setPointerCapture(event.pointerId);
      this.draw();
      return;
    }

    this.selectedId = null;
    this.onSelection(null);
    this.draw();
  };

  private readonly onPointerMove = (event: PointerEvent): void => {
    if (this.drag === null) return;
    const point = this.eventPoint(event);
    const drag = this.drag;
    let next = this.boxes;

    if (drag.kind === 'move') {
      next = this.replaceBox(drag.boxId, {
        ...drag.original,
        centerX: drag.original.centerX + point.x - drag.start.x,
        centerY: drag.original.centerY + point.y - drag.start.y,
      });
    } else if (drag.kind === 'resize') {
      const angle = drag.original.angleDeg * Math.PI / 180;
      const axisX = { x: Math.cos(angle), y: Math.sin(angle) };
      const axisY = { x: -Math.sin(angle), y: Math.cos(angle) };
      const delta = {
        x: point.x - drag.anchor.x,
        y: point.y - drag.anchor.y,
      };
      const projectedX = delta.x * axisX.x + delta.y * axisX.y;
      const projectedY = delta.x * axisY.x + delta.y * axisY.y;
      const width = Math.max(2, drag.signX * projectedX);
      const height = Math.max(2, drag.signY * projectedY);
      const draggedCorner = {
        x: drag.anchor.x + axisX.x * drag.signX * width + axisY.x * drag.signY * height,
        y: drag.anchor.y + axisX.y * drag.signX * width + axisY.y * drag.signY * height,
      };
      next = this.replaceBox(drag.boxId, {
        ...drag.original,
        centerX: (drag.anchor.x + draggedCorner.x) / 2,
        centerY: (drag.anchor.y + draggedCorner.y) / 2,
        width,
        height,
      });
    } else if (drag.kind === 'rotate') {
      let angle = Math.atan2(
        point.y - drag.original.centerY,
        point.x - drag.original.centerX,
      ) * 180 / Math.PI + 90;
      if (event.shiftKey) angle = Math.round(angle / 5) * 5;
      next = this.replaceBox(drag.boxId, {
        ...drag.original,
        angleDeg: normalizeAngle(angle),
      });
    } else {
      const width = Math.max(2, Math.abs(point.x - drag.start.x));
      const height = Math.max(2, Math.abs(point.y - drag.start.y));
      next = this.replaceBox(drag.boxId, {
        id: drag.boxId,
        centerX: (drag.start.x + point.x) / 2,
        centerY: (drag.start.y + point.y) / 2,
        width,
        height,
        angleDeg: 0,
      });
    }

    this.boxes = next;
    this.onChange(this.copyBoxes());
    this.onSelection(this.selectedBox());
    this.draw();
  };

  private readonly onPointerUp = (event: PointerEvent): void => {
    if (this.drag === null) return;
    if (this.canvas.hasPointerCapture(event.pointerId)) this.canvas.releasePointerCapture(event.pointerId);
    const createdId = this.drag.kind === 'create' ? this.drag.boxId : null;
    this.drag = null;
    if (createdId !== null) {
      const created = this.boxes.find((box) => box.id === createdId);
      if (created !== undefined && (created.width < 4 || created.height < 4)) {
        this.boxes = this.boxes.filter((box) => box.id !== createdId);
        this.selectedId = null;
        this.onSelection(null);
      }
    }
    this.onChange(this.copyBoxes());
    this.draw();
  };

  private replaceBox(boxId: string, replacement: AnnotationBox): AnnotationBox[] {
    return this.boxes.map((box) => box.id === boxId ? replacement : box);
  }

  private commit(boxes: AnnotationBox[]): void {
    this.boxes = boxes.map((box) => ({ ...box }));
    this.onChange(this.copyBoxes());
    this.draw();
  }

  private copyBoxes(): AnnotationBox[] {
    return this.boxes.map((box) => ({ ...box }));
  }

  private eventPoint(event: PointerEvent): Point {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * this.canvas.width / rect.width,
      y: (event.clientY - rect.top) * this.canvas.height / rect.height,
    };
  }

  private displayScale(): number {
    const rect = this.canvas.getBoundingClientRect();
    return rect.width <= 0 || this.canvas.width <= 0 ? 1 : rect.width / this.canvas.width;
  }

  private draw(): void {
    const image = this.image;
    if (image === null) return;
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.context.drawImage(image, 0, 0);
    const scale = this.displayScale();
    const lineWidth = Math.max(1, 2 / scale);
    const handleRadius = 6 / scale;
    const rotationOffset = 28 / scale;

    for (const box of this.boxes) {
      const selected = box.id === this.selectedId;
      const label = this.labels.get(box.id);
      const points = corners(box);
      this.context.save();
      this.context.lineWidth = selected ? lineWidth * 1.8 : lineWidth;
      this.context.strokeStyle = label?.tentative === true ? '#ff6b6b' : selected ? '#ffd43b' : '#57e389';
      this.context.beginPath();
      this.context.moveTo(points[0]?.x ?? 0, points[0]?.y ?? 0);
      for (let index = 1; index < points.length; index += 1) {
        const point = points[index];
        if (point !== undefined) this.context.lineTo(point.x, point.y);
      }
      this.context.closePath();
      this.context.stroke();
      this.context.restore();

      if (label !== undefined) this.drawLabel(box, label.text, label.tentative, scale);
      if (selected) {
        for (const point of points) {
          this.context.beginPath();
          this.context.fillStyle = '#ffd43b';
          this.context.arc(point.x, point.y, handleRadius, 0, Math.PI * 2);
          this.context.fill();
          this.context.strokeStyle = '#111318';
          this.context.lineWidth = Math.max(1, 1 / scale);
          this.context.stroke();
        }
        const top = {
          x: ((points[0]?.x ?? 0) + (points[1]?.x ?? 0)) / 2,
          y: ((points[0]?.y ?? 0) + (points[1]?.y ?? 0)) / 2,
        };
        const rotate = rotationHandle(box, rotationOffset);
        this.context.beginPath();
        this.context.strokeStyle = '#ffd43b';
        this.context.lineWidth = lineWidth;
        this.context.moveTo(top.x, top.y);
        this.context.lineTo(rotate.x, rotate.y);
        this.context.stroke();
        this.context.beginPath();
        this.context.fillStyle = '#ffd43b';
        this.context.arc(rotate.x, rotate.y, handleRadius * 1.1, 0, Math.PI * 2);
        this.context.fill();
      }
    }
  }

  private drawLabel(
    box: AnnotationBox,
    text: string,
    tentative: boolean,
    scale: number,
  ): void {
    const points = corners(box);
    const left = Math.min(...points.map((point) => point.x));
    const top = Math.min(...points.map((point) => point.y));
    const fontSize = 14 / scale;
    const paddingX = 5 / scale;
    const paddingY = 3 / scale;
    const labelText = tentative ? `仮 ${text}` : text;
    this.context.save();
    this.context.font = `700 ${fontSize}px ui-monospace, SFMono-Regular, Consolas, monospace`;
    const width = this.context.measureText(labelText).width + paddingX * 2;
    const height = fontSize + paddingY * 2;
    const y = Math.max(0, top - height);
    this.context.fillStyle = tentative ? 'rgba(112, 28, 34, 0.92)' : 'rgba(11, 57, 35, 0.92)';
    this.context.fillRect(left, y, width, height);
    this.context.fillStyle = '#ffffff';
    this.context.textBaseline = 'top';
    this.context.fillText(labelText, left + paddingX, y + paddingY);
    this.context.restore();
  }
}
