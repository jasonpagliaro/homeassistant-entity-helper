import React, { act } from "react";
import { createRoot, Root } from "react-dom/client";
import { FlowEditorApp, type FlowEditorConfig } from "./main";

class ResizeObserverMock {
  constructor(private readonly callback: ResizeObserverCallback) {}

  observe(): void {
    this.callback([], this as unknown as ResizeObserver);
  }

  disconnect(): void {}

  unobserve(): void {}
}

const TEST_CANVAS_WIDTH = 800;
const TEST_CANVAS_HEIGHT = 600;

const TEST_CONFIG: FlowEditorConfig = {
  editorId: "flow-editor-test",
  pageKind: "config",
  readOnly: false,
  flowVariableKey: "_haev_flow",
  automationDocument: {
    alias: "Wheel Test",
    description: "",
    trigger: [{ platform: "time", at: "19:00:00" }],
    condition: [],
    action: [{ service: "light.turn_on", target: { entity_id: "light.kitchen" } }],
    mode: "single",
  },
  catalogs: {
    entities: [],
    services: [],
    automations: [],
    warnings: [],
  },
};

describe("FlowEditorApp wheel gestures", () => {
  const originalActEnvironment = (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean })
    .IS_REACT_ACT_ENVIRONMENT;
  const originalResizeObserver = globalThis.ResizeObserver;
  const originalClientWidth = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientWidth");
  const originalClientHeight = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientHeight");
  const originalGetBoundingClientRect = HTMLElement.prototype.getBoundingClientRect;

  let host: HTMLDivElement;
  let root: Root | null = null;

  beforeAll(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    globalThis.ResizeObserver = ResizeObserverMock as typeof ResizeObserver;
    Object.defineProperty(HTMLElement.prototype, "clientWidth", {
      configurable: true,
      get() {
        return this instanceof HTMLElement && this.dataset.flowEditorCanvas === "true"
          ? TEST_CANVAS_WIDTH
          : 0;
      },
    });
    Object.defineProperty(HTMLElement.prototype, "clientHeight", {
      configurable: true,
      get() {
        return this instanceof HTMLElement && this.dataset.flowEditorCanvas === "true"
          ? TEST_CANVAS_HEIGHT
          : 0;
      },
    });
    HTMLElement.prototype.getBoundingClientRect = function getBoundingClientRect(): DOMRect {
      if (this instanceof HTMLElement && this.dataset.flowEditorCanvas === "true") {
        return DOMRect.fromRect({
          x: 0,
          y: 0,
          width: TEST_CANVAS_WIDTH,
          height: TEST_CANVAS_HEIGHT,
        });
      }
      return originalGetBoundingClientRect.call(this);
    };
  });

  afterAll(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
      originalActEnvironment;
    globalThis.ResizeObserver = originalResizeObserver;
    if (originalClientWidth) {
      Object.defineProperty(HTMLElement.prototype, "clientWidth", originalClientWidth);
    }
    if (originalClientHeight) {
      Object.defineProperty(HTMLElement.prototype, "clientHeight", originalClientHeight);
    }
    HTMLElement.prototype.getBoundingClientRect = originalGetBoundingClientRect;
  });

  beforeEach(() => {
    host = document.createElement("div");
    document.body.appendChild(host);
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    root = null;
    host.remove();
  });

  function renderEditor(): HTMLDivElement {
    root = createRoot(host);
    act(() => {
      root?.render(React.createElement(FlowEditorApp, { config: TEST_CONFIG }));
    });

    const canvas = host.querySelector<HTMLDivElement>("[data-flow-editor-canvas='true']");
    if (!canvas) {
      throw new Error("Expected flow editor canvas to render.");
    }
    return canvas;
  }

  function readZoom(): string {
    const readout = host.querySelector(".flow-editor__zoom-readout");
    if (!readout) {
      throw new Error("Expected zoom readout to render.");
    }
    return readout.textContent ?? "";
  }

  it("ignores plain wheel input so page scrolling can continue", () => {
    const canvas = renderEditor();
    const initialZoom = readZoom();
    const event = new WheelEvent("wheel", {
      deltaY: -120,
      clientX: 200,
      clientY: 180,
      bubbles: true,
      cancelable: true,
    });

    act(() => {
      canvas.dispatchEvent(event);
    });

    expect(event.defaultPrevented).toBe(false);
    expect(readZoom()).toBe(initialZoom);
  });

  it("zooms and cancels default scrolling for ctrl/cmd wheel input", () => {
    for (const modifier of [{ ctrlKey: true }, { metaKey: true }]) {
      const canvas = renderEditor();
      const initialZoom = readZoom();
      const event = new WheelEvent("wheel", {
        deltaY: -120,
        clientX: 200,
        clientY: 180,
        bubbles: true,
        cancelable: true,
        ...modifier,
      });

      act(() => {
        canvas.dispatchEvent(event);
      });

      expect(event.defaultPrevented).toBe(true);
      expect(readZoom()).not.toBe(initialZoom);

      act(() => {
        root?.unmount();
      });
      root = null;
      host.innerHTML = "";
    }
  });
});
