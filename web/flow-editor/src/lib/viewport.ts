import { FlowNode, FlowViewport } from "./flow-model";

export const FLOW_EDITOR_NODE_WIDTH = 180;
export const FLOW_EDITOR_NODE_HEIGHT = 82;
export const FLOW_EDITOR_STAGE_PADDING = 48;
export const FLOW_EDITOR_MIN_ZOOM = 0.35;
export const FLOW_EDITOR_MAX_ZOOM = 2;
export const FLOW_EDITOR_ZOOM_STEP = 0.1;

const DEFAULT_VIEWPORT: FlowViewport = { x: 0, y: 0, zoom: 1 };

export interface ViewportSize {
  width: number;
  height: number;
}

export interface ViewportPoint {
  x: number;
  y: number;
}

export interface StageBounds {
  minX: number;
  minY: number;
  width: number;
  height: number;
}

function roundViewportValue(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function sanitizeDimension(value: number, fallback: number): number {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function centerOffset(viewportSize: number, scaledSize: number): number {
  return roundViewportValue((viewportSize - scaledSize) / 2);
}

function clampOffset(offset: number, viewportSize: number, scaledSize: number): number {
  if (scaledSize <= viewportSize) {
    return centerOffset(viewportSize, scaledSize);
  }
  if (!Number.isFinite(offset)) {
    return 0;
  }
  return roundViewportValue(Math.min(0, Math.max(viewportSize - scaledSize, offset)));
}

export function clampZoom(zoom: number): number {
  if (!Number.isFinite(zoom)) {
    return DEFAULT_VIEWPORT.zoom;
  }
  return roundViewportValue(Math.min(FLOW_EDITOR_MAX_ZOOM, Math.max(FLOW_EDITOR_MIN_ZOOM, zoom)));
}

export function areViewportsEqual(left: FlowViewport, right: FlowViewport): boolean {
  return left.x === right.x && left.y === right.y && left.zoom === right.zoom;
}

export function hasSavedViewport(viewport: Partial<FlowViewport> | null | undefined): boolean {
  if (!viewport) {
    return false;
  }
  return (
    (typeof viewport.x === "number" && roundViewportValue(viewport.x) !== DEFAULT_VIEWPORT.x) ||
    (typeof viewport.y === "number" && roundViewportValue(viewport.y) !== DEFAULT_VIEWPORT.y) ||
    (typeof viewport.zoom === "number" && clampZoom(viewport.zoom) !== DEFAULT_VIEWPORT.zoom)
  );
}

export function computeStageBounds(nodes: Array<Pick<FlowNode, "position">>): StageBounds {
  if (nodes.length === 0) {
    return {
      minX: 0,
      minY: 0,
      width: FLOW_EDITOR_NODE_WIDTH + FLOW_EDITOR_STAGE_PADDING * 2,
      height: FLOW_EDITOR_NODE_HEIGHT + FLOW_EDITOR_STAGE_PADDING * 2,
    };
  }

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (const node of nodes) {
    minX = Math.min(minX, node.position.x);
    minY = Math.min(minY, node.position.y);
    maxX = Math.max(maxX, node.position.x + FLOW_EDITOR_NODE_WIDTH);
    maxY = Math.max(maxY, node.position.y + FLOW_EDITOR_NODE_HEIGHT);
  }

  return {
    minX: minX - FLOW_EDITOR_STAGE_PADDING,
    minY: minY - FLOW_EDITOR_STAGE_PADDING,
    width: sanitizeDimension(maxX - minX + FLOW_EDITOR_STAGE_PADDING * 2, FLOW_EDITOR_NODE_WIDTH),
    height: sanitizeDimension(maxY - minY + FLOW_EDITOR_STAGE_PADDING * 2, FLOW_EDITOR_NODE_HEIGHT),
  };
}

export function clampViewport(
  viewport: FlowViewport,
  stageBounds: Pick<StageBounds, "width" | "height">,
  canvasSize: ViewportSize,
): FlowViewport {
  const zoom = clampZoom(viewport.zoom);
  const stageWidth = sanitizeDimension(
    stageBounds.width,
    FLOW_EDITOR_NODE_WIDTH + FLOW_EDITOR_STAGE_PADDING * 2,
  );
  const stageHeight = sanitizeDimension(
    stageBounds.height,
    FLOW_EDITOR_NODE_HEIGHT + FLOW_EDITOR_STAGE_PADDING * 2,
  );
  const viewportWidth = sanitizeDimension(canvasSize.width, stageWidth);
  const viewportHeight = sanitizeDimension(canvasSize.height, stageHeight);
  const scaledStageWidth = stageWidth * zoom;
  const scaledStageHeight = stageHeight * zoom;

  return {
    x: clampOffset(viewport.x, viewportWidth, scaledStageWidth),
    y: clampOffset(viewport.y, viewportHeight, scaledStageHeight),
    zoom,
  };
}

export function fitViewportToStage(
  stageBounds: Pick<StageBounds, "width" | "height">,
  canvasSize: ViewportSize,
): FlowViewport {
  if (canvasSize.width <= 0 || canvasSize.height <= 0) {
    return DEFAULT_VIEWPORT;
  }

  const stageWidth = sanitizeDimension(stageBounds.width, FLOW_EDITOR_NODE_WIDTH);
  const stageHeight = sanitizeDimension(stageBounds.height, FLOW_EDITOR_NODE_HEIGHT);
  const zoom = clampZoom(Math.min(canvasSize.width / stageWidth, canvasSize.height / stageHeight));

  return clampViewport(
    {
      x: (canvasSize.width - stageWidth * zoom) / 2,
      y: (canvasSize.height - stageHeight * zoom) / 2,
      zoom,
    },
    stageBounds,
    canvasSize,
  );
}

export function zoomViewportAtPoint(
  viewport: FlowViewport,
  nextZoom: number,
  point: ViewportPoint,
  stageBounds: Pick<StageBounds, "width" | "height">,
  canvasSize: ViewportSize,
): FlowViewport {
  const currentZoom = clampZoom(viewport.zoom);
  const zoom = clampZoom(nextZoom);
  const graphX = (point.x - viewport.x) / currentZoom;
  const graphY = (point.y - viewport.y) / currentZoom;

  return clampViewport(
    {
      x: point.x - graphX * zoom,
      y: point.y - graphY * zoom,
      zoom,
    },
    stageBounds,
    canvasSize,
  );
}
