import React, { useEffect, useState } from "react";
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
  importAutomationDocument,
  normalizeAutomationDocument,
  prepareAutomationDocumentForSave,
  removeAtPath,
  updateAtPath,
} from "./lib/flow-model";

type EditorCatalogs = {
  entities: Array<{ entity_id: string; friendly_name?: string; domain?: string }>;
  services: Array<{ service_id: string; name?: string; description?: string }>;
  automations: Array<{ entity_id: string; name?: string; config_key?: string }>;
  warnings: string[];
};

type FlowEditorConfig = {
  editorId: string;
  pageKind: string;
  readOnly: boolean;
  flowVariableKey: string;
  automationDocument: Record<string, unknown>;
  catalogs: EditorCatalogs;
  saveFormId?: string | null;
  yamlTextareaId?: string | null;
};

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

function FlowEditorApp({ config }: { config: FlowEditorConfig }) {
  const [automationDocument, setAutomationDocument] = useState<Record<string, unknown>>(() =>
    normalizeAutomationDocument(config.automationDocument),
  );
  const graph = importAutomationDocument(
    normalizeAutomationDocument(automationDocument),
    config.flowVariableKey,
  );
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(graph.nodes[0]?.id ?? null);

  useEffect(() => {
    if (!graph.nodes.some((node) => node.id === selectedNodeId)) {
      setSelectedNodeId(graph.nodes[0]?.id ?? null);
    }
    updateYamlTextarea(config, automationDocument, graph);
  }, [automationDocument, selectedNodeId]);

  const selectedNode = graph.nodes.find((node) => node.id === selectedNodeId);

  const addTrigger = () =>
    setAutomationDocument((current) => appendToArrayPath(current, ["trigger"], createDefaultTrigger()));
  const addCondition = () =>
    setAutomationDocument((current) =>
      appendToArrayPath(current, ["condition"], createDefaultCondition()),
    );
  const addAction = (kind: FlowNodeKind) =>
    setAutomationDocument((current) => appendToArrayPath(current, ["action"], createDefaultAction(kind)));
  const ensureVariables = () =>
    setAutomationDocument((current) => updateAtPath(current, ["variables"], (current as any).variables ?? {}));

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
              onClick={() => submitFlowForm(config, automationDocument, graph)}
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

      <div className="flow-editor__layout">
        <div className="flow-editor__canvas">
          <svg className="flow-editor__edges" viewBox="0 0 1200 1200" preserveAspectRatio="xMinYMin slice">
            {graph.edges.map((edge) => {
              const source = graph.nodes.find((node) => node.id === edge.source);
              const target = graph.nodes.find((node) => node.id === edge.target);
              if (!source || !target) {
                return null;
              }
              const x1 = source.position.x + 180;
              const y1 = source.position.y + 40;
              const x2 = target.position.x;
              const y2 = target.position.y + 40;
              return (
                <g key={edge.id}>
                  <path
                    d={`M ${x1} ${y1} C ${x1 + 60} ${y1}, ${x2 - 60} ${y2}, ${x2} ${y2}`}
                    className="flow-editor__edge-path"
                  />
                  {edge.label ? (
                    <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 6} className="flow-editor__edge-label">
                      {edge.label}
                    </text>
                  ) : null}
                </g>
              );
            })}
          </svg>

          {graph.nodes.map((node) => (
            <button
              key={node.id}
              type="button"
              className={`flow-editor__node flow-editor__node--${node.kind}${selectedNodeId === node.id ? " is-selected" : ""}${node.locked ? " is-locked" : ""}`}
              style={{ left: `${node.position.x}px`, top: `${node.position.y}px` }}
              onClick={() => setSelectedNodeId(node.id)}
            >
              <span className="flow-editor__node-title">{node.title}</span>
              <span className="flow-editor__node-subtitle">{node.subtitle}</span>
            </button>
          ))}
        </div>

        <NodeEditor
          config={config}
          node={selectedNode}
          automationDocument={automationDocument}
          setAutomationDocument={setAutomationDocument}
          graph={graph}
        />
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
