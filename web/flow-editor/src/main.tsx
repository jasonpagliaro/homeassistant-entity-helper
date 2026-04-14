import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { stringify as stringifyYaml } from "yaml";
import "./styles.css";
import {
  appendToArrayPath,
  createDefaultAction,
  createDefaultCondition,
  createDefaultTrigger,
  FlowNode,
  FlowNodeKind,
  FlowViewport,
  importAutomationDocument,
  normalizeAutomationDocument,
  prepareAutomationDocumentForSave,
  removeAtPath,
  updateAtPath,
} from "./lib/flow-model";
import {
  areViewportsEqual,
  clampViewport,
  computeStageBounds,
  fitViewportToStage,
  FLOW_EDITOR_ZOOM_STEP,
  hasSavedViewport,
  ViewportSize,
  zoomViewportAtPoint,
} from "./lib/viewport";
import {
  getStageEdgeLayout,
  getStageNodeLayout,
  getStageRenderMetrics,
} from "./lib/canvas-layout";

export type EditorCatalogs = {
  entities: Array<{ entity_id: string; friendly_name?: string; domain?: string }>;
  services: Array<{ service_id: string; name?: string; description?: string }>;
  automations: Array<{ entity_id: string; name?: string; config_key?: string }>;
  warnings: string[];
};

export type FlowEditorConfig = {
  editorId: string;
  pageKind: string;
  readOnly: boolean;
  flowVariableKey: string;
  automationDocument: Record<string, unknown>;
  catalogs: EditorCatalogs;
  saveFormId?: string | null;
  yamlTextareaId?: string | null;
};

type PointerPanState = {
  pointerId: number;
  startClientX: number;
  startClientY: number;
  startX: number;
  startY: number;
  moved: boolean;
};

const DRAG_THRESHOLD_PX = 6;

function parseConfig(elementId: string): FlowEditorConfig {
  const script = document.getElementById(elementId);
  if (!(script instanceof HTMLScriptElement)) {
    throw new Error(`Missing flow editor config script '${elementId}'.`);
  }
  return JSON.parse(script.textContent || "{}") as FlowEditorConfig;
}

function stringifyJson(value: unknown): string {
  return JSON.stringify(value ?? {}, null, 2);
}

function updateYamlTextarea(
  config: FlowEditorConfig,
  automationDocument: Record<string, unknown>,
  graph: ReturnType<typeof importAutomationDocument>,
): void {
  if (!config.yamlTextareaId) {
    return;
  }
  const textarea = document.getElementById(config.yamlTextareaId);
  if (!(textarea instanceof HTMLTextAreaElement)) {
    return;
  }
  const prepared = prepareAutomationDocumentForSave(
    normalizeAutomationDocument(automationDocument),
    graph,
    config.flowVariableKey,
  );
  textarea.value = stringifyYaml(prepared);
}

function submitFlowForm(
  config: FlowEditorConfig,
  automationDocument: Record<string, unknown>,
  graph: ReturnType<typeof importAutomationDocument>,
): void {
  if (!config.saveFormId) {
    return;
  }
  const form = document.getElementById(config.saveFormId);
  if (!(form instanceof HTMLFormElement)) {
    return;
  }
  const automationJson = form.querySelector<HTMLInputElement>('input[name="automation_json"]');
  const editOrigin = form.querySelector<HTMLInputElement>('input[name="edit_origin"]');
  const prepared = prepareAutomationDocumentForSave(
    normalizeAutomationDocument(automationDocument),
    graph,
    config.flowVariableKey,
  );

  if (automationJson) {
    automationJson.value = JSON.stringify(prepared);
  }
  if (editOrigin) {
    editOrigin.value = "flow_edit";
  }
  updateYamlTextarea(config, automationDocument, graph);
  form.dataset.flowSubmitPending = "true";
  form.requestSubmit();
}

function bootPageControls(): void {
  document.querySelectorAll<HTMLFormElement>('[data-flow-yaml-form="true"]').forEach((form) => {
    form.addEventListener("submit", () => {
      if (form.dataset.flowSubmitPending === "true") {
        form.dataset.flowSubmitPending = "false";
        return;
      }
      const automationJson = form.querySelector<HTMLInputElement>('input[name="automation_json"]');
      const editOrigin = form.querySelector<HTMLInputElement>('input[name="edit_origin"]');
      if (automationJson) {
        automationJson.value = "";
      }
      if (editOrigin) {
        editOrigin.value = "manual_edit";
      }
    });
  });

  document.querySelectorAll<HTMLElement>("[data-flow-toggle-group]").forEach((group) => {
    const buttons = Array.from(group.querySelectorAll<HTMLButtonElement>("[data-flow-toggle-target]"));
    const activate = (targetId: string) => {
      document.querySelectorAll<HTMLElement>("[data-flow-toggle-panel]").forEach((panel) => {
        if (!panel.id) {
          return;
        }
        panel.hidden = panel.id !== targetId;
      });
      buttons.forEach((button) => {
        button.classList.toggle("is-active", button.dataset.flowToggleTarget === targetId);
      });
    };
    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        if (button.dataset.flowToggleTarget) {
          activate(button.dataset.flowToggleTarget);
        }
      });
    });
  });
}

function NodeEditor({
  config,
  node,
  automationDocument,
  setAutomationDocument,
  graph,
}: {
  config: FlowEditorConfig;
  node: FlowNode | undefined;
  automationDocument: Record<string, unknown>;
  setAutomationDocument: React.Dispatch<React.SetStateAction<Record<string, unknown>>>;
  graph: ReturnType<typeof importAutomationDocument>;
}) {
  const [draftJson, setDraftJson] = useState("");
  const [parseError, setParseError] = useState("");

  useEffect(() => {
    setDraftJson(stringifyJson(node?.payload ?? {}));
    setParseError("");
  }, [node?.id]);

  if (!node) {
    return <div className="flow-editor__empty">Select a node to inspect or edit it.</div>;
  }

  const applyDraft = () => {
    try {
      const parsed = JSON.parse(draftJson);
      setAutomationDocument((current) => updateAtPath(current, node.path, parsed));
      setParseError("");
    } catch (error) {
      setParseError(error instanceof Error ? error.message : String(error));
    }
  };

  const removeNode = () => {
    if (config.readOnly || !node.removable) {
      return;
    }
    setAutomationDocument((current) => removeAtPath(current, node.path));
  };

  const updateServiceValue = (value: string) => {
    const current =
      node.payload && typeof node.payload === "object"
        ? { ...(node.payload as Record<string, unknown>) }
        : {};
    current.service = value;
    if ("action" in current) {
      delete current.action;
    }
    setAutomationDocument((root) => updateAtPath(root, node.path, current));
  };

  const updateDelayValue = (value: string) => {
    const current =
      node.payload && typeof node.payload === "object"
        ? { ...(node.payload as Record<string, unknown>) }
        : {};
    current.delay = value;
    setAutomationDocument((root) => updateAtPath(root, node.path, current));
  };

  return (
    <div className="flow-editor__inspector">
      <div className="flow-editor__inspector-header">
        <div>
          <h4>{node.title}</h4>
          <p>{node.subtitle}</p>
        </div>
        {node.removable && !config.readOnly ? (
          <button type="button" className="button-secondary danger" onClick={removeNode}>
            Remove
          </button>
        ) : null}
      </div>

      {node.kind === "action_service" ? (
        <label>
          Service
          <input
            type="text"
            list={`${config.editorId}-services`}
            value={String((node.payload as Record<string, unknown>)?.service ?? "")}
            onChange={(event) => updateServiceValue(event.target.value)}
            disabled={config.readOnly}
          />
        </label>
      ) : null}

      {node.kind === "action_delay" ? (
        <label>
          Delay
          <input
            type="text"
            value={String((node.payload as Record<string, unknown>)?.delay ?? "")}
            onChange={(event) => updateDelayValue(event.target.value)}
            disabled={config.readOnly}
          />
        </label>
      ) : null}

      <label>
        JSON
        <textarea
          rows={14}
          value={draftJson}
          onChange={(event) => setDraftJson(event.target.value)}
          readOnly={config.readOnly || node.locked}
        />
      </label>
      {parseError ? <div className="flash flash-error">{parseError}</div> : null}
      {!config.readOnly && !node.locked ? (
        <div className="flow-editor__inspector-actions">
          <button type="button" onClick={applyDraft}>
            Apply JSON Changes
          </button>
          <button
            type="button"
            className="button-secondary"
            onClick={() => submitFlowForm(config, automationDocument, graph)}
          >
            Save Flow Changes
          </button>
        </div>
      ) : null}
      {node.locked ? (
        <p className="flow-editor__note">
          This node is locked because it uses an unsupported structure. Save keeps it intact.
        </p>
      ) : null}
    </div>
  );
}

export function FlowEditorApp({ config }: { config: FlowEditorConfig }) {
  const [automationDocument, setAutomationDocument] = useState<Record<string, unknown>>(() =>
    normalizeAutomationDocument(config.automationDocument),
  );
  const graph = importAutomationDocument(
    normalizeAutomationDocument(automationDocument),
    config.flowVariableKey,
  );
  const stageBounds = computeStageBounds(graph.nodes);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(graph.nodes[0]?.id ?? null);
  const [viewport, setViewport] = useState<FlowViewport>(graph.viewport);
  const [canvasSize, setCanvasSize] = useState<ViewportSize>({ width: 0, height: 0 });
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const panStateRef = useRef<PointerPanState | null>(null);
  const suppressNodeClickRef = useRef(false);
  const suppressResetTimerRef = useRef<number | null>(null);
  const viewportInitializedRef = useRef(false);
  const graphWithViewport = { ...graph, viewport };
  const selectedNode = graph.nodes.find((node) => node.id === selectedNodeId);
  const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));
  const stageMetrics = getStageRenderMetrics(stageBounds, viewport.zoom);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return undefined;
    }

    const updateSize = () => {
      setCanvasSize({
        width: canvas.clientWidth,
        height: canvas.clientHeight,
      });
    };

    updateSize();

    let observer: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(updateSize);
      observer.observe(canvas);
    }
    window.addEventListener("resize", updateSize);

    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", updateSize);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (suppressResetTimerRef.current !== null) {
        window.clearTimeout(suppressResetTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!graph.nodes.some((node) => node.id === selectedNodeId)) {
      setSelectedNodeId(graph.nodes[0]?.id ?? null);
    }
  }, [graph, selectedNodeId]);

  useEffect(() => {
    if (canvasSize.width <= 0 || canvasSize.height <= 0) {
      return;
    }

    setViewport((current) => {
      const next = !viewportInitializedRef.current
        ? hasSavedViewport(graph.viewport)
          ? clampViewport(graph.viewport, stageBounds, canvasSize)
          : fitViewportToStage(stageBounds, canvasSize)
        : clampViewport(current, stageBounds, canvasSize);
      viewportInitializedRef.current = true;
      return areViewportsEqual(current, next) ? current : next;
    });
  }, [
    canvasSize.height,
    canvasSize.width,
    graph.viewport.x,
    graph.viewport.y,
    graph.viewport.zoom,
    stageBounds.height,
    stageBounds.width,
  ]);

  useEffect(() => {
    updateYamlTextarea(config, automationDocument, graphWithViewport);
  }, [automationDocument, config, graphWithViewport]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return undefined;
    }

    const handleCanvasWheel = (event: WheelEvent) => {
      if (!event.ctrlKey && !event.metaKey) {
        return;
      }
      if (canvasSize.width <= 0 || canvasSize.height <= 0 || event.deltaY === 0) {
        return;
      }

      const canvasRect = canvas.getBoundingClientRect();
      const anchor = {
        x: event.clientX - canvasRect.left,
        y: event.clientY - canvasRect.top,
      };

      event.preventDefault();
      event.stopPropagation();
      setViewport((current) => {
        const next = zoomViewportAtPoint(
          current,
          current.zoom + (event.deltaY < 0 ? FLOW_EDITOR_ZOOM_STEP : -FLOW_EDITOR_ZOOM_STEP),
          anchor,
          stageBounds,
          canvasSize,
        );
        return areViewportsEqual(current, next) ? current : next;
      });
    };

    canvas.addEventListener("wheel", handleCanvasWheel, { passive: false });
    return () => {
      canvas.removeEventListener("wheel", handleCanvasWheel);
    };
  }, [
    canvasSize.height,
    canvasSize.width,
    stageBounds.height,
    stageBounds.minX,
    stageBounds.minY,
    stageBounds.width,
  ]);

  const queueSuppressReset = () => {
    if (suppressResetTimerRef.current !== null) {
      window.clearTimeout(suppressResetTimerRef.current);
    }
    suppressResetTimerRef.current = window.setTimeout(() => {
      suppressNodeClickRef.current = false;
      suppressResetTimerRef.current = null;
    }, 0);
  };

  const updateViewport = (nextBuilder: (current: FlowViewport) => FlowViewport) => {
    setViewport((current) => {
      const next = nextBuilder(current);
      return areViewportsEqual(current, next) ? current : next;
    });
  };

  const stopPanning = () => {
    const activePan = panStateRef.current;
    panStateRef.current = null;
    setIsPanning(false);
    if (activePan?.moved) {
      queueSuppressReset();
    }
  };

  const addTrigger = () =>
    setAutomationDocument((current) => appendToArrayPath(current, ["trigger"], createDefaultTrigger()));
  const addCondition = () =>
    setAutomationDocument((current) =>
      appendToArrayPath(current, ["condition"], createDefaultCondition()),
    );
  const addAction = (kind: FlowNodeKind) =>
    setAutomationDocument((current) => appendToArrayPath(current, ["action"], createDefaultAction(kind)));
  const ensureVariables = () =>
    setAutomationDocument((current) =>
      updateAtPath(current, ["variables"], (current as Record<string, unknown>).variables ?? {}),
    );

  const handleCanvasPointerDownCapture = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 2) {
      return;
    }
    panStateRef.current = {
      pointerId: event.pointerId,
      startClientX: event.clientX,
      startClientY: event.clientY,
      startX: viewport.x,
      startY: viewport.y,
      moved: false,
    };
    suppressNodeClickRef.current = false;
    event.currentTarget.setPointerCapture(event.pointerId);
    setIsPanning(true);
    event.preventDefault();
  };

  const handleCanvasPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const activePan = panStateRef.current;
    if (!activePan || activePan.pointerId !== event.pointerId) {
      return;
    }

    const deltaX = event.clientX - activePan.startClientX;
    const deltaY = event.clientY - activePan.startClientY;
    if (!activePan.moved && Math.hypot(deltaX, deltaY) >= DRAG_THRESHOLD_PX) {
      activePan.moved = true;
      suppressNodeClickRef.current = true;
    }

    updateViewport((current) =>
      clampViewport(
        {
          x: activePan.startX + deltaX,
          y: activePan.startY + deltaY,
          zoom: current.zoom,
        },
        stageBounds,
        canvasSize,
      ),
    );
    event.preventDefault();
  };

  const handleCanvasPointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    if (panStateRef.current?.pointerId !== event.pointerId) {
      return;
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    stopPanning();
  };

  const handleCanvasPointerCancel = (event: React.PointerEvent<HTMLDivElement>) => {
    if (panStateRef.current?.pointerId !== event.pointerId) {
      return;
    }
    stopPanning();
  };

  const fitCanvas = () => {
    if (canvasSize.width <= 0 || canvasSize.height <= 0) {
      return;
    }
    updateViewport(() => fitViewportToStage(stageBounds, canvasSize));
  };

  const zoomCanvas = (delta: number) => {
    if (canvasSize.width <= 0 || canvasSize.height <= 0) {
      return;
    }
    const center = { x: canvasSize.width / 2, y: canvasSize.height / 2 };
    updateViewport((current) =>
      zoomViewportAtPoint(current, current.zoom + delta, center, stageBounds, canvasSize),
    );
  };

  const resetZoom = () => {
    if (canvasSize.width <= 0 || canvasSize.height <= 0) {
      return;
    }
    const center = { x: canvasSize.width / 2, y: canvasSize.height / 2 };
    updateViewport((current) => zoomViewportAtPoint(current, 1, center, stageBounds, canvasSize));
  };

  const handleNodeSelect = (nodeId: string) => {
    if (suppressNodeClickRef.current) {
      return;
    }
    setSelectedNodeId(nodeId);
  };

  return (
    <div className="flow-editor">
      <div className="flow-editor__toolbar">
        <label>
          Alias
          <input
            type="text"
            value={String((automationDocument as Record<string, unknown>).alias ?? "")}
            onChange={(event) =>
              setAutomationDocument((current) => updateAtPath(current, ["alias"], event.target.value))
            }
            disabled={config.readOnly}
          />
        </label>
        <label className="flow-editor__toolbar-wide">
          Description
          <input
            type="text"
            value={String((automationDocument as Record<string, unknown>).description ?? "")}
            onChange={(event) =>
              setAutomationDocument((current) =>
                updateAtPath(current, ["description"], event.target.value),
              )
            }
            disabled={config.readOnly}
          />
        </label>
        <label>
          Mode
          <input
            type="text"
            value={String((automationDocument as Record<string, unknown>).mode ?? "single")}
            onChange={(event) =>
              setAutomationDocument((current) => updateAtPath(current, ["mode"], event.target.value))
            }
            disabled={config.readOnly}
          />
        </label>
      </div>

      {!config.readOnly ? (
        <div className="flow-editor__toolbar flow-editor__toolbar-actions">
          <button type="button" onClick={addTrigger}>Add Trigger</button>
          <button type="button" onClick={addCondition}>Add Condition</button>
          <button type="button" onClick={() => addAction("action_service")}>Add Service</button>
          <button type="button" onClick={() => addAction("action_delay")}>Add Delay</button>
          <button type="button" onClick={() => addAction("action_choose")}>Add Choose</button>
          <button type="button" onClick={() => addAction("action_repeat")}>Add Repeat</button>
          <button type="button" onClick={() => addAction("action_wait_for_trigger")}>Add Wait</button>
          {graph.nodes.some((node) => node.kind === "variables") ? null : (
            <button type="button" onClick={ensureVariables}>Add Variables</button>
          )}
          {config.saveFormId ? (
            <button
              type="button"
              className="button-secondary"
              onClick={() => submitFlowForm(config, automationDocument, graphWithViewport)}
            >
              Save Flow Changes
            </button>
          ) : null}
        </div>
      ) : null}

      {config.catalogs.warnings.length > 0 ? (
        <div className="flash flash-warning">{config.catalogs.warnings.join(" ")}</div>
      ) : null}
      {graph.warnings.length > 0 ? (
        <div className="flash flash-warning">{graph.warnings.join(" ")}</div>
      ) : null}

      <div className={`flow-editor__layout${inspectorCollapsed ? " is-inspector-collapsed" : ""}`}>
        <div className={`flow-editor__canvas-shell${isPanning ? " is-panning" : ""}`}>
          <div className="flow-editor__canvas-toolbar">
            <div className="flow-editor__canvas-toolbar-group">
              <button type="button" className="button-secondary" onClick={fitCanvas}>
                Fit
              </button>
              <button type="button" className="button-secondary" onClick={resetZoom}>
                100%
              </button>
              <button type="button" className="button-secondary" onClick={() => zoomCanvas(-FLOW_EDITOR_ZOOM_STEP)}>
                -
              </button>
              <button type="button" className="button-secondary" onClick={() => zoomCanvas(FLOW_EDITOR_ZOOM_STEP)}>
                +
              </button>
              <span className="flow-editor__zoom-readout">{Math.round(viewport.zoom * 100)}%</span>
            </div>
            <button
              type="button"
              className="button-secondary"
              onClick={() => setInspectorCollapsed((current) => !current)}
            >
              {inspectorCollapsed ? "Show Inspector" : "Hide Inspector"}
            </button>
          </div>

          <div
            ref={canvasRef}
            className="flow-editor__canvas"
            data-flow-editor-canvas="true"
            onContextMenu={(event) => event.preventDefault()}
            onPointerDownCapture={handleCanvasPointerDownCapture}
            onPointerMove={handleCanvasPointerMove}
            onPointerUp={handleCanvasPointerUp}
            onPointerCancel={handleCanvasPointerCancel}
            onLostPointerCapture={stopPanning}
          >
            <div
              className="flow-editor__stage"
              style={{
                width: stageMetrics.stageWidth,
                height: stageMetrics.stageHeight,
                left: viewport.x,
                top: viewport.y,
              }}
            >
              <svg
                className="flow-editor__edges"
                viewBox={`0 0 ${stageMetrics.stageWidth} ${stageMetrics.stageHeight}`}
                preserveAspectRatio="none"
              >
                {graph.edges.map((edge) => {
                  const source = nodesById.get(edge.source);
                  const target = nodesById.get(edge.target);
                  if (!source || !target) {
                    return null;
                  }
                  const layout = getStageEdgeLayout(source, target, stageBounds, stageMetrics);
                  return (
                    <g key={edge.id}>
                      <path
                        d={layout.path}
                        className="flow-editor__edge-path"
                        style={{ strokeWidth: layout.strokeWidth }}
                      />
                      {edge.label ? (
                        <text
                          x={layout.labelX}
                          y={layout.labelY}
                          className="flow-editor__edge-label"
                          style={{ fontSize: layout.labelFontSize }}
                        >
                          {edge.label}
                        </text>
                      ) : null}
                    </g>
                  );
                })}
              </svg>

              {graph.nodes.map((node) => {
                const layout = getStageNodeLayout(node, stageBounds, stageMetrics);
                return (
                  <button
                    key={node.id}
                    type="button"
                    className={`flow-editor__node flow-editor__node--${node.kind}${selectedNodeId === node.id ? " is-selected" : ""}${node.locked ? " is-locked" : ""}`}
                    style={{
                      left: layout.left,
                      top: layout.top,
                      width: layout.width,
                      minHeight: layout.minHeight,
                      padding: layout.padding,
                      gap: layout.gap,
                      borderRadius: layout.borderRadius,
                    }}
                    onClick={() => handleNodeSelect(node.id)}
                  >
                    <span className="flow-editor__node-title" style={{ fontSize: layout.titleFontSize }}>
                      {node.title}
                    </span>
                    <span className="flow-editor__node-subtitle" style={{ fontSize: layout.subtitleFontSize }}>
                      {node.subtitle}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {inspectorCollapsed ? null : (
          <NodeEditor
            config={config}
            node={selectedNode}
            automationDocument={automationDocument}
            setAutomationDocument={setAutomationDocument}
            graph={graphWithViewport}
          />
        )}
      </div>

      <datalist id={`${config.editorId}-services`}>
        {config.catalogs.services.map((service) => (
          <option key={service.service_id} value={service.service_id}>
            {service.name || service.service_id}
          </option>
        ))}
      </datalist>
    </div>
  );
}

bootPageControls();

document.querySelectorAll<HTMLElement>("[data-flow-editor-root='true']").forEach((element) => {
  const configId = element.dataset.flowEditorConfigId;
  if (!configId) {
    return;
  }
  const config = parseConfig(configId);
  createRoot(element).render(<FlowEditorApp config={config} />);
});
