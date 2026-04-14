import {
  getStageEdgeLayout,
  getStageNodeLayout,
  getStageRenderMetrics,
} from "./canvas-layout";

describe("canvas-layout", () => {
  it("keeps base geometry at 100% zoom", () => {
    const metrics = getStageRenderMetrics({ width: 520, height: 360 }, 1);

    expect(metrics).toMatchObject({
      stageWidth: 520,
      stageHeight: 360,
      nodeWidth: 180,
      nodeHeight: 82,
      nodePadding: 12,
      nodeGap: 4,
      nodeRadius: 14,
      nodeTitleSize: 14,
      nodeSubtitleSize: 12,
      edgeStrokeWidth: 2.5,
    });
  });

  it("scales node geometry down below 100% zoom", () => {
    const metrics = getStageRenderMetrics({ width: 520, height: 360 }, 0.5);
    const layout = getStageNodeLayout(
      { position: { x: 72, y: 44 } },
      { minX: 12, minY: 4 },
      metrics,
    );

    expect(layout).toEqual({
      left: 30,
      top: 20,
      width: 90,
      minHeight: 41,
      padding: 6,
      gap: 2,
      borderRadius: 7,
      titleFontSize: 7,
      subtitleFontSize: 6,
    });
  });

  it("scales edge geometry up above 100% zoom", () => {
    const metrics = getStageRenderMetrics({ width: 520, height: 360 }, 1.25);
    const layout = getStageEdgeLayout(
      { position: { x: 40, y: 24 } },
      { position: { x: 320, y: 180 } },
      { minX: 0, minY: 0 },
      metrics,
    );

    expect(layout).toEqual({
      path: "M 275 81.25 C 350 81.25, 325 276.25, 400 276.25",
      labelX: 337.5,
      labelY: 171.25,
      labelFontSize: 15,
      strokeWidth: 3.125,
    });
  });
});
