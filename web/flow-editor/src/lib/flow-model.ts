export type PathSegment = string | number;

export type AutomationDocument = Record<string, unknown> & {
  alias: string;
  description?: string;
  trigger: unknown[];
  condition?: unknown[];
  action: unknown[];
  mode?: string;
  variables?: Record<string, unknown>;
};

export type FlowNodeKind =
  | "variables"
  | "trigger"
  | "condition"
  | "action_service"
  | "action_delay"
  | "action_choose"
  | "action_repeat"
  | "action_wait_for_trigger"
  | "raw";

export type FlowSection = "meta" | "trigger" | "condition" | "action";

export interface FlowNode {
  id: string;
  kind: FlowNodeKind;
  title: string;
  subtitle: string;
  path: PathSegment[];
  payload: unknown;
  position: { x: number; y: number };
  section: FlowSection;
  locked: boolean;
  removable: boolean;
  parentId?: string;
}

export interface FlowEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
}

export interface FlowViewport {
  x: number;
  y: number;
  zoom: number;
}

export interface FlowGraph {
  nodes: FlowNode[];
  edges: FlowEdge[];
  warnings: string[];
  viewport: FlowViewport;
}

type SavedFlowMetadata = {
  version?: number;
  nodes?: Array<{ id?: string; x?: number; y?: number }>;
  edges?: Array<{ id?: string; source?: string; target?: string; label?: string }>;
  viewport?: Partial<FlowViewport>;
  import_warnings?: string[];
};

const DEFAULT_VIEWPORT: FlowViewport = { x: 0, y: 0, zoom: 1 };

function cloneValue<T>(value: T): T {
  return JSON.parse(JSON.stringify(value));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sanitizeVariables(value: unknown): Record<string, unknown> {
  if (!isRecord(value)) {
    return {};
  }
  return cloneValue(value);
}

export function normalizeAutomationDocument(value: unknown): AutomationDocument {
  if (!isRecord(value)) {
    throw new Error("Automation document must be an object.");
  }
  const document = cloneValue(value) as AutomationDocument;
  const alias = String(document.alias ?? "").trim();
  if (!alias) {
    throw new Error("Automation alias is required.");
  }

  const trigger = Array.isArray(document.trigger)
    ? document.trigger
    : Array.isArray((document as Record<string, unknown>).triggers)
      ? ((document as Record<string, unknown>).triggers as unknown[])
      : null;
  const condition = Array.isArray(document.condition)
    ? document.condition
    : Array.isArray((document as Record<string, unknown>).conditions)
      ? ((document as Record<string, unknown>).conditions as unknown[])
      : [];
  const action = Array.isArray(document.action)
    ? document.action
    : Array.isArray((document as Record<string, unknown>).actions)
      ? ((document as Record<string, unknown>).actions as unknown[])
      : null;

  if (!trigger || trigger.length === 0) {
    throw new Error("Automation requires at least one trigger.");
  }
  if (!action || action.length === 0) {
    throw new Error("Automation requires at least one action.");
  }

  document.alias = alias;
  document.description = String(document.description ?? "").trim();
  document.trigger = trigger;
  document.condition = Array.isArray(condition) ? condition : [];
  document.action = action;
  document.mode = String(document.mode ?? "single").trim() || "single";

  if ("triggers" in document) {
    delete (document as Record<string, unknown>).triggers;
  }
  if ("conditions" in document) {
    delete (document as Record<string, unknown>).conditions;
  }
  if ("actions" in document) {
    delete (document as Record<string, unknown>).actions;
  }
  if (document.variables !== undefined && !isRecord(document.variables)) {
    throw new Error("Automation variables must be an object when present.");
  }
  return document;
}

function nodeIdFromPath(path: PathSegment[]): string {
  return path.map((segment) => String(segment).replace(/[^a-zA-Z0-9_-]+/g, "_")).join("__") || "root";
}

function readSavedMetadata(document: AutomationDocument, flowVariableKey: string): SavedFlowMetadata | null {
  const variables = sanitizeVariables(document.variables);
  const candidate = variables[flowVariableKey];
  if (!isRecord(candidate)) {
    return null;
  }
  return candidate as SavedFlowMetadata;
}

function summarizeArrayCount(label: string, value: unknown): string {
  return Array.isArray(value) ? `${value.length} ${label}` : `0 ${label}`;
}

function summarizeTrigger(trigger: unknown): string {
  if (!isRecord(trigger)) {
    return "Unsupported trigger";
  }
  const platform = String(trigger.platform ?? "trigger");
  const entityId = String(trigger.entity_id ?? "").trim();
  if (platform === "time") {
    return String(trigger.at ?? "time");
  }
  if (entityId) {
    return `${platform}: ${entityId}`;
  }
  return platform;
}

function summarizeCondition(condition: unknown): string {
  if (Array.isArray(condition)) {
    return summarizeArrayCount("conditions", condition);
  }
  if (!isRecord(condition)) {
    return "Unsupported condition";
  }
  const kind = String(condition.condition ?? "condition");
  const entityId = String(condition.entity_id ?? "").trim();
  return entityId ? `${kind}: ${entityId}` : kind;
}

function summarizeAction(action: unknown): { kind: FlowNodeKind; title: string; subtitle: string; locked: boolean } {
  if (!isRecord(action)) {
    return { kind: "raw", title: "Raw Action", subtitle: "Unsupported action payload", locked: true };
  }
  if ("service" in action || "action" in action) {
    const serviceId = String(action.service ?? action.action ?? "service");
    return { kind: "action_service", title: "Service Action", subtitle: serviceId, locked: false };
  }
  if ("delay" in action) {
    return { kind: "action_delay", title: "Delay", subtitle: String(action.delay ?? "delay"), locked: false };
  }
  if ("choose" in action) {
    return {
      kind: "action_choose",
      title: "Choose",
      subtitle: summarizeArrayCount("branches", action.choose),
      locked: false,
    };
  }
  if ("repeat" in action) {
    const repeatConfig = isRecord(action.repeat) ? action.repeat : {};
    return {
      kind: "action_repeat",
      title: "Repeat",
      subtitle: summarizeArrayCount("steps", repeatConfig.sequence),
      locked: false,
    };
  }
  if ("wait_for_trigger" in action) {
    return {
      kind: "action_wait_for_trigger",
      title: "Wait For Trigger",
      subtitle: summarizeArrayCount("triggers", action.wait_for_trigger),
      locked: false,
    };
  }
  if ("variables" in action) {
    return {
      kind: "variables",
      title: "Variables",
      subtitle: `${Object.keys(sanitizeVariables(action.variables)).length} keys`,
      locked: false,
    };
  }
  return { kind: "raw", title: "Raw Action", subtitle: "Unsupported action payload", locked: true };
}

function createPositionResolver(savedMetadata: SavedFlowMetadata | null) {
  const savedPositions = new Map<string, { x: number; y: number }>();
  for (const item of savedMetadata?.nodes ?? []) {
    if (!item || typeof item.id !== "string") {
      continue;
    }
    savedPositions.set(item.id, {
      x: typeof item.x === "number" ? item.x : 0,
      y: typeof item.y === "number" ? item.y : 0,
    });
  }
  const rowsByColumn = new Map<number, number>();
  return (id: string, column: number): { x: number; y: number } => {
    const saved = savedPositions.get(id);
    if (saved) {
      return saved;
    }
    const row = rowsByColumn.get(column) ?? 0;
    rowsByColumn.set(column, row + 1);
    return {
      x: 48 + column * 280,
      y: 48 + row * 140,
    };
  };
}

function getAtPath(root: unknown, path: PathSegment[]): unknown {
  let current = root;
  for (const segment of path) {
    if (typeof segment === "number") {
      if (!Array.isArray(current)) {
        return undefined;
      }
      current = current[segment];
      continue;
    }
    if (!isRecord(current)) {
      return undefined;
    }
    current = current[segment];
  }
  return current;
}

export function updateAtPath<T>(root: T, path: PathSegment[], nextValue: unknown): T {
  if (path.length === 0) {
    return cloneValue(nextValue) as T;
  }
  const cloned = cloneValue(root) as unknown;
  let current: unknown = cloned;
  for (let index = 0; index < path.length - 1; index += 1) {
    const segment = path[index];
    if (typeof segment === "number") {
      if (!Array.isArray(current)) {
        throw new Error("Expected array while updating flow path.");
      }
      current = current[segment];
      continue;
    }
    if (!isRecord(current)) {
      throw new Error("Expected object while updating flow path.");
    }
    current = current[segment];
  }
  const last = path[path.length - 1];
  if (typeof last === "number") {
    if (!Array.isArray(current)) {
      throw new Error("Expected array while updating flow item.");
    }
    current[last] = cloneValue(nextValue);
  } else {
    if (!isRecord(current)) {
      throw new Error("Expected object while updating flow field.");
    }
    current[last] = cloneValue(nextValue);
  }
  return cloned as T;
}

export function removeAtPath<T>(root: T, path: PathSegment[]): T {
  if (path.length === 0) {
    throw new Error("Cannot remove the root automation document.");
  }
  const cloned = cloneValue(root) as unknown;
  let current: unknown = cloned;
  for (let index = 0; index < path.length - 1; index += 1) {
    const segment = path[index];
    if (typeof segment === "number") {
      if (!Array.isArray(current)) {
        throw new Error("Expected array while removing flow item.");
      }
      current = current[segment];
      continue;
    }
    if (!isRecord(current)) {
      throw new Error("Expected object while removing flow item.");
    }
    current = current[segment];
  }
  const last = path[path.length - 1];
  if (typeof last === "number") {
    if (!Array.isArray(current)) {
      throw new Error("Expected array while removing flow item.");
    }
    current.splice(last, 1);
  } else {
    if (!isRecord(current)) {
      throw new Error("Expected object while removing flow field.");
    }
    delete current[last];
  }
  return cloned as T;
}

export function appendToArrayPath<T>(root: T, path: PathSegment[], nextValue: unknown): T {
  const existing = getAtPath(root, path);
  if (!Array.isArray(existing)) {
    throw new Error("Expected an array while appending flow item.");
  }
  const appended = cloneValue(existing);
  appended.push(cloneValue(nextValue));
  return updateAtPath(root, path, appended);
}

function addEdge(edges: FlowEdge[], source: string, target: string, label?: string): void {
  edges.push({
    id: `${source}->${target}${label ? `:${label}` : ""}`,
    source,
    target,
    label,
  });
}

export function importAutomationDocument(input: AutomationDocument, flowVariableKey: string): FlowGraph {
  const document = normalizeAutomationDocument(input);
  const savedMetadata = readSavedMetadata(document, flowVariableKey);
  const nextPosition = createPositionResolver(savedMetadata);
  const nodes: FlowNode[] = [];
  const edges: FlowEdge[] = [];
  const warnings = [...(savedMetadata?.import_warnings ?? [])];

  const variables = sanitizeVariables(document.variables);
  if (flowVariableKey in variables) {
    delete variables[flowVariableKey];
  }
  if (Object.keys(variables).length > 0 || savedMetadata !== null) {
    const id = "variables";
    nodes.push({
      id,
      kind: "variables",
      title: "Variables",
      subtitle: `${Object.keys(variables).length} keys`,
      path: ["variables"],
      payload: variables,
      position: nextPosition(id, 0),
      section: "meta",
      locked: false,
      removable: true,
    });
  }

  let previousTriggerId: string | null = null;
  for (let index = 0; index < document.trigger.length; index += 1) {
    const item = document.trigger[index];
    const path: PathSegment[] = ["trigger", index];
    const id = nodeIdFromPath(path);
    const locked = !isRecord(item);
    nodes.push({
      id,
      kind: "trigger",
      title: `Trigger ${index + 1}`,
      subtitle: summarizeTrigger(item),
      path,
      payload: item,
      position: nextPosition(id, 0),
      section: "trigger",
      locked,
      removable: true,
    });
    if (locked) {
      warnings.push(`Trigger ${index + 1} is not a standard trigger object and may require YAML review.`);
    }
    if (previousTriggerId) {
      addEdge(edges, previousTriggerId, id);
    }
    previousTriggerId = id;
  }

  let previousConditionId: string | null = null;
  for (let index = 0; index < document.condition.length; index += 1) {
    const item = document.condition[index];
    const path: PathSegment[] = ["condition", index];
    const id = nodeIdFromPath(path);
    const locked = !isRecord(item);
    nodes.push({
      id,
      kind: "condition",
      title: `Condition ${index + 1}`,
      subtitle: summarizeCondition(item),
      path,
      payload: item,
      position: nextPosition(id, 1),
      section: "condition",
      locked,
      removable: true,
    });
    if (locked) {
      warnings.push(`Condition ${index + 1} is not a standard condition object and may require YAML review.`);
    }
    if (previousConditionId) {
      addEdge(edges, previousConditionId, id);
    }
    previousConditionId = id;
  }

  function importActionSequence(
    sequence: unknown[],
    pathPrefix: PathSegment[],
    column: number,
    parentId?: string,
    parentLabel?: string,
  ): { firstId?: string; lastId?: string } {
    let firstId: string | undefined;
    let previousId: string | undefined;
    for (let index = 0; index < sequence.length; index += 1) {
      const item = sequence[index];
      const path = [...pathPrefix, index];
      const id = nodeIdFromPath(path);
      const summary = summarizeAction(item);
      nodes.push({
        id,
        kind: summary.kind,
        title: summary.title,
        subtitle: summary.subtitle,
        path,
        payload: item,
        position: nextPosition(id, column),
        section: "action",
        locked: summary.locked,
        removable: true,
        parentId,
      });
      if (summary.kind === "raw") {
        warnings.push(`Unsupported action at ${path.join(".")} is shown as a locked raw node.`);
      }
      if (!firstId) {
        firstId = id;
      }
      if (previousId) {
        addEdge(edges, previousId, id);
      } else if (parentId) {
        addEdge(edges, parentId, id, parentLabel);
      }
      previousId = id;

      if (summary.kind === "action_choose" && isRecord(item) && Array.isArray(item.choose)) {
        item.choose.forEach((choice, choiceIndex) => {
          if (!isRecord(choice)) {
            return;
          }
          const branchConditionsPath = [...path, "choose", choiceIndex, "conditions"];
          const branchConditionsId = nodeIdFromPath(branchConditionsPath);
          const branchConditions = Array.isArray(choice.conditions) ? choice.conditions : [];
          nodes.push({
            id: branchConditionsId,
            kind: "condition",
            title: `Choice ${choiceIndex + 1} Conditions`,
            subtitle: summarizeCondition(branchConditions),
            path: branchConditionsPath,
            payload: branchConditions,
            position: nextPosition(branchConditionsId, column + 1),
            section: "condition",
            locked: false,
            removable: false,
            parentId: id,
          });
          addEdge(edges, id, branchConditionsId, `Choice ${choiceIndex + 1}`);
          const branchSequence = Array.isArray(choice.sequence) ? choice.sequence : [];
          importActionSequence(
            branchSequence,
            [...path, "choose", choiceIndex, "sequence"],
            column + 2,
            branchConditionsId,
          );
        });
        if (Array.isArray(item.default)) {
          importActionSequence(item.default, [...path, "default"], column + 2, id, "Default");
        }
      }

      if (summary.kind === "action_repeat" && isRecord(item)) {
        const repeatConfig = isRecord(item.repeat) ? item.repeat : {};
        const repeatSequence = Array.isArray(repeatConfig.sequence) ? repeatConfig.sequence : [];
        importActionSequence(repeatSequence, [...path, "repeat", "sequence"], column + 1, id, "Repeat");
      }
    }
    return { firstId, lastId: previousId };
  }

  const actionGraph = importActionSequence(document.action, ["action"], 2);
  if (previousConditionId && actionGraph.firstId) {
    addEdge(edges, previousConditionId, actionGraph.firstId);
  } else if (previousTriggerId && actionGraph.firstId) {
    addEdge(edges, previousTriggerId, actionGraph.firstId);
  }

  return {
    nodes,
    edges,
    warnings,
    viewport: {
      x: typeof savedMetadata?.viewport?.x === "number" ? savedMetadata.viewport.x : DEFAULT_VIEWPORT.x,
      y: typeof savedMetadata?.viewport?.y === "number" ? savedMetadata.viewport.y : DEFAULT_VIEWPORT.y,
      zoom: typeof savedMetadata?.viewport?.zoom === "number" ? savedMetadata.viewport.zoom : DEFAULT_VIEWPORT.zoom,
    },
  };
}

export function buildFlowMetadata(graph: FlowGraph): SavedFlowMetadata {
  return {
    version: 1,
    nodes: graph.nodes.map((node) => ({
      id: node.id,
      x: node.position.x,
      y: node.position.y,
    })),
    edges: graph.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.label,
    })),
    viewport: graph.viewport,
    import_warnings: graph.warnings,
  };
}

export function prepareAutomationDocumentForSave(
  input: AutomationDocument,
  graph: FlowGraph,
  flowVariableKey: string,
): AutomationDocument {
  const document = normalizeAutomationDocument(input);
  const variables = sanitizeVariables(document.variables);
  variables[flowVariableKey] = buildFlowMetadata(graph);
  document.variables = variables;
  return document;
}

export function createDefaultTrigger(): Record<string, unknown> {
  return {
    platform: "state",
    entity_id: "",
    to: "on",
  };
}

export function createDefaultCondition(): Record<string, unknown> {
  return {
    condition: "state",
    entity_id: "",
    state: "on",
  };
}

export function createDefaultAction(kind: FlowNodeKind): Record<string, unknown> {
  if (kind === "action_delay") {
    return { delay: "00:05:00" };
  }
  if (kind === "action_choose") {
    return {
      choose: [{ conditions: [], sequence: [] }],
      default: [],
    };
  }
  if (kind === "action_repeat") {
    return {
      repeat: {
        count: 2,
        sequence: [],
      },
    };
  }
  if (kind === "action_wait_for_trigger") {
    return {
      wait_for_trigger: [createDefaultTrigger()],
      timeout: "00:05:00",
    };
  }
  return {
    service: "",
    target: {
      entity_id: "",
    },
    data: {},
  };
}
