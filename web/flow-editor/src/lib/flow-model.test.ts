import {
  buildFlowMetadata,
  importAutomationDocument,
  normalizeAutomationDocument,
  prepareAutomationDocumentForSave,
} from "./flow-model";

describe("flow-model", () => {
  it("preserves flow metadata in variables", () => {
    const document = normalizeAutomationDocument({
      alias: "Evening Mode",
      description: "",
      trigger: [{ platform: "time", at: "19:00:00" }],
      condition: [],
      action: [{ service: "light.turn_on", target: { entity_id: "light.kitchen" } }],
      mode: "single",
      variables: {
        threshold: 3,
      },
    });

    const graph = importAutomationDocument(document, "_haev_flow");
    const saved = prepareAutomationDocumentForSave(document, graph, "_haev_flow");

    expect(saved.variables).toHaveProperty("threshold", 3);
    expect(saved.variables).toHaveProperty("_haev_flow");
    expect((saved.variables as Record<string, unknown>)._haev_flow).toMatchObject({
      version: 1,
    });
  });

  it("imports unsupported actions as locked raw nodes", () => {
    const graph = importAutomationDocument(
      normalizeAutomationDocument({
        alias: "Unsupported",
        description: "",
        trigger: [{ platform: "state", entity_id: "binary_sensor.door", to: "on" }],
        condition: [],
        action: [{ fire_event: "custom_thing" }],
        mode: "single",
      }),
      "_haev_flow",
    );

    expect(graph.nodes.some((node) => node.kind === "raw" && node.locked)).toBe(true);
    expect(graph.warnings.some((warning) => warning.includes("Unsupported action"))).toBe(true);
  });

  it("round-trips saved positions from metadata", () => {
    const document = normalizeAutomationDocument({
      alias: "Positioned",
      description: "",
      trigger: [{ platform: "time", at: "19:00:00" }],
      condition: [],
      action: [{ service: "light.turn_on", target: { entity_id: "light.kitchen" } }],
      mode: "single",
    });
    const graph = importAutomationDocument(document, "_haev_flow");
    const nextMetadata = buildFlowMetadata({
      ...graph,
      nodes: graph.nodes.map((node) =>
        node.id === "action__0" ? { ...node, position: { x: 999, y: 321 } } : node,
      ),
    });
    const saved = normalizeAutomationDocument({
      ...document,
      variables: {
        _haev_flow: nextMetadata,
      },
    });

    const reimported = importAutomationDocument(saved, "_haev_flow");
    expect(reimported.nodes.find((node) => node.id === "action__0")?.position).toEqual({
      x: 999,
      y: 321,
    });
  });
});
