import type { AnnotationBox } from './types';

export interface Point {
  x: number;
  y: number;
}

export function radians(angleDeg: number): number {
  return angleDeg * Math.PI / 180;
}

export function normalizeAngle(angleDeg: number): number {
  let angle = angleDeg % 360;
  if (angle >= 180) angle -= 360;
  if (angle < -180) angle += 360;
  return angle;
}

export function localToWorld(box: AnnotationBox, local: Point): Point {
  const angle = radians(box.angleDeg);
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return {
    x: box.centerX + local.x * cosine - local.y * sine,
    y: box.centerY + local.x * sine + local.y * cosine,
  };
}

export function worldToLocal(box: AnnotationBox, point: Point): Point {
  const angle = radians(-box.angleDeg);
  const dx = point.x - box.centerX;
  const dy = point.y - box.centerY;
  return {
    x: dx * Math.cos(angle) - dy * Math.sin(angle),
    y: dx * Math.sin(angle) + dy * Math.cos(angle),
  };
}

export function corners(box: AnnotationBox): Point[] {
  const halfWidth = box.width / 2;
  const halfHeight = box.height / 2;
  return [
    localToWorld(box, { x: -halfWidth, y: -halfHeight }),
    localToWorld(box, { x: halfWidth, y: -halfHeight }),
    localToWorld(box, { x: halfWidth, y: halfHeight }),
    localToWorld(box, { x: -halfWidth, y: halfHeight }),
  ];
}

export function rotationHandle(box: AnnotationBox, offset: number): Point {
  return localToWorld(box, { x: 0, y: -box.height / 2 - offset });
}

export function containsPoint(box: AnnotationBox, point: Point): boolean {
  const local = worldToLocal(box, point);
  return Math.abs(local.x) <= box.width / 2 && Math.abs(local.y) <= box.height / 2;
}

export function distance(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function boxInside(box: AnnotationBox, width: number, height: number): boolean {
  return corners(box).every((point) => (
    point.x >= -0.001
    && point.y >= -0.001
    && point.x <= width + 0.001
    && point.y <= height + 0.001
  ));
}

export function splitBox(
  box: AnnotationBox,
  axis: 'x' | 'y',
  gap = 2,
): [AnnotationBox, AnnotationBox] {
  if (axis === 'x') {
    const childWidth = Math.max(2, (box.width - gap) / 2);
    const offset = (childWidth + gap) / 2;
    const firstCenter = localToWorld(box, { x: -offset, y: 0 });
    const secondCenter = localToWorld(box, { x: offset, y: 0 });
    return [
      { ...box, id: crypto.randomUUID(), centerX: firstCenter.x, centerY: firstCenter.y, width: childWidth },
      { ...box, id: crypto.randomUUID(), centerX: secondCenter.x, centerY: secondCenter.y, width: childWidth },
    ];
  }
  const childHeight = Math.max(2, (box.height - gap) / 2);
  const offset = (childHeight + gap) / 2;
  const firstCenter = localToWorld(box, { x: 0, y: -offset });
  const secondCenter = localToWorld(box, { x: 0, y: offset });
  return [
    { ...box, id: crypto.randomUUID(), centerX: firstCenter.x, centerY: firstCenter.y, height: childHeight },
    { ...box, id: crypto.randomUUID(), centerX: secondCenter.x, centerY: secondCenter.y, height: childHeight },
  ];
}
