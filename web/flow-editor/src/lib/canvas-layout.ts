import { FlowNode } from "./flow-model";
import {
  FLOW_EDITOR_NODE_HEIGHT,
  FLOW_EDITOR_NODE_WIDTH,
  StageBounds,
} from "./viewport";

const FLOW_EDITOR_NODE_PADDING = 12;
const FLOW_EDITOR_NODE_GAP = 4;
const FLOW_EDITOR_NODE_RADIUS = 14;
const FLOW_EDITOR_NODE_TITLE_SIZE = 14;
const FLOW_EDITOR_NODE_SUBTITLE_SIZE = 12;
const FLOW_EDITOR_EDGE_CURVE_OFFSET = 60;
const FLOW_EDITOR_EDGE_LABEL_OFFSET = 6;
const FLOW_EDITOR_EDGE_STROKE_WIDTH = 2.5;

function scaleValue(value: number, zoom: number): number {
  return Math.round(value * zoom * 1000) / 1000;
}

export interface StageRenderMetrics {
  zoom: number;
  stageWidth: number;
  stageHeight: number;
  nodeWidth: number;
  nodeHeight: number;
  nodePadding: number;
  nodeGap: number;
  nodeRadius: number;
  nodeTitleSize: number;
  nodeSubtitleSize: number;
  edgeCurveOffset: number;
  edgeLabelOffset: number;
  edgeStrokeWidth: number;
}

export interface StageNodeLayout {
  left: number;
  top: number;
  width: number;
  minHeight: number;
  padding: number;
  gap: number;
  borderRadius: number;
  titleFontSize: number;
  subtitleFontSize: number;
}

export interface StageEdgeLayout {
  path: string;
  labelX: number;
  labelY: number;
  labelFontSize: number;
  strokeWidth: number;
}

export function getStageRenderMetrics(stageBounds: Pick<StageBounds, "width" | "height">, zoom: number): StageRenderMetrics {
  return {
    zoom,
    stageWidth: scaleValue(stageBounds.width, zoom),
    stageHeight: scaleValue(stageBounds.height, zoom),
    nodeWidth: scaleValue(FLOW_EDITOR_NODE_WIDTH, zoom),
    nodeHeight: scaleValue(FLOW_EDITOR_NODE_HEIGHT, zoom),
    nodePadding: scaleValue(FLOW_EDITOR_NODE_PADDING, zoom),
    nodeGap: scaleValue(FLOW_EDITOR_NODE_GAP, zoom),
    nodeRadius: scaleValue(FLOW_EDITOR_NODE_RADIUS, zoom),
    nodeTitleSize: scaleValue(FLOW_EDITOR_NODE_TITLE_SIZE, zoom),
    nodeSubtitleSize: scaleValue(FLOW_EDITOR_NODE_SUBTITLE_SIZE, zoom),
    edgeCurveOffset: scaleValue(FLOW_EDITOR_EDGE_CURVE_OFFSET, zoom),
    edgeLabelOffset: scaleValue(FLOW_EDITOR_EDGE_LABEL_OFFSET, zoom),
    edgeStrokeWidth: Math.max(1, scaleValue(FLOW_EDITOR_EDGE_STROKE_WIDTH, zoom)),
  };
}

export function getStageNodeLayout(
  node: Pick<FlowNode, "position">,
  stageBounds: Pick<StageBounds, "minX" | "minY">,
  metrics: StageRenderMetrics,
): StageNodeLayout {
  return {
    left: scaleValue(node.position.x - stageBounds.minX, metrics.zoom),
    top: scaleValue(node.position.y - stageBounds.minY, metrics.zoom),
    width: metrics.nodeWidth,
    minHeight: metrics.nodeHeight,
    padding: metrics.nodePadding,
    gap: metrics.nodeGap,
    borderRadius: metrics.nodeRadius,
    titleFontSize: metrics.nodeTitleSize,
    subtitleFontSize: metrics.nodeSubtitleSize,
  };
}

export function getStageEdgeLayout(
  source: Pick<FlowNode, "position">,
  target: Pick<FlowNode, "position">,
  stageBounds: Pick<StageBounds, "minX" | "minY">,
  metrics: StageRenderMetrics,
): StageEdgeLayout {
  const x1 = scaleValue(source.position.x - stageBounds.minX, metrics.zoom) + metrics.nodeWidth;
  const y1 = scaleValue(source.position.y - stageBounds.minY, metrics.zoom) + metrics.nodeHeight / 2;
  const x2 = scaleValue(target.position.x - stageBounds.minX, metrics.zoom);
  const y2 = scaleValue(target.position.y - stageBounds.minY, metrics.zoom) + metrics.nodeHeight / 2;

  return {
    path: `M ${x1} ${y1} C ${x1 + metrics.edgeCurveOffset} ${y1}, ${x2 - metrics.edgeCurveOffset} ${y2}, ${x2} ${y2}`,
    labelX: (x1 + x2) / 2,
    labelY: (y1 + y2) / 2 - metrics.edgeLabelOffset,
    labelFontSize: metrics.nodeSubtitleSize,
    strokeWidth: metrics.edgeStrokeWidth,
  };
}
