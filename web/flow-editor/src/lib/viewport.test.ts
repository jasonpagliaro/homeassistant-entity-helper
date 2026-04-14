import {
  clampViewport,
  computeStageBounds,
  fitViewportToStage,
  zoomViewportAtPoint,
} from "./viewport";

describe("viewport", () => {
  it("fits the full stage inside the canvas", () => {
    expect(
      fitViewportToStage(
        { width: 800, height: 600 },
        { width: 400, height: 300 },
      ),
    ).toEqual({
      x: 0,
      y: 0,
      zoom: 0.5,
    });
  });

  it("keeps the pointer anchor stable while zooming", () => {
    const initialViewport = fitViewportToStage(
      { width: 800, height: 600 },
      { width: 400, height: 300 },
    );

    expect(
      zoomViewportAtPoint(
        initialViewport,
        1,
        { x: 200, y: 150 },
        { width: 800, height: 600 },
        { width: 400, height: 300 },
      ),
    ).toEqual({
      x: -200,
      y: -150,
      zoom: 1,
    });
  });

  it("clamps panning so large stages stay in bounds", () => {
    expect(
      clampViewport(
        { x: 40, y: -500, zoom: 1 },
        { width: 1000, height: 500 },
        { width: 400, height: 300 },
      ),
    ).toEqual({
      x: 0,
      y: -200,
      zoom: 1,
    });
  });

  it("recenters smaller stages while clamping zoom", () => {
    expect(
      clampViewport(
        { x: 10, y: -40, zoom: 5 },
        { width: 200, height: 100 },
        { width: 500, height: 300 },
      ),
    ).toEqual({
      x: 50,
      y: 50,
      zoom: 2,
    });
  });

  it("computes stage bounds with outer padding", () => {
    expect(
      computeStageBounds([
        { position: { x: -30, y: 20 } },
        { position: { x: 320, y: 240 } },
      ]),
    ).toEqual({
      minX: -78,
      minY: -28,
      width: 626,
      height: 398,
    });
  });
});
