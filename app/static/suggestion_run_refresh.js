(function () {
  const root = document.querySelector("[data-suggestion-refresh]");
  if (!(root instanceof HTMLElement)) {
    return;
  }

  const mode = (root.getAttribute("data-suggestion-refresh") || "").trim().toLowerCase();
  if (mode !== "list" && mode !== "detail") {
    return;
  }

  const profileIdRaw = (root.getAttribute("data-profile-id") || "").trim();
  const profileId = Number(profileIdRaw);
  if (!Number.isInteger(profileId) || profileId <= 0) {
    return;
  }

  const TERMINAL_STATUSES = new Set(["succeeded", "failed"]);
  const SUCCESS_INTERVAL_VISIBLE_MS = 2500;
  const SUCCESS_INTERVAL_HIDDEN_MS = 10000;
  const FAILURE_INTERVALS_MS = [5000, 10000, 20000, 30000];

  let timerId = null;
  let stopped = false;
  let reloadRequested = false;
  let hasPendingReload = false;
  let failureCount = 0;

  const runStateById = new Map();
  const refreshBanner = root.querySelector("[data-refresh-banner]");
  const refreshNowButton = root.querySelector("[data-refresh-now]");

  function isTerminalStatus(status) {
    return TERMINAL_STATUSES.has(status);
  }

  function clearTimer() {
    if (timerId !== null) {
      window.clearTimeout(timerId);
      timerId = null;
    }
  }

  function schedule(delayMs) {
    if (stopped || reloadRequested) {
      return;
    }
    clearTimer();
    timerId = window.setTimeout(runPollCycle, delayMs);
  }

  function activeSuccessDelayMs() {
    return document.visibilityState === "hidden" ? SUCCESS_INTERVAL_HIDDEN_MS : SUCCESS_INTERVAL_VISIBLE_MS;
  }

  function parseRunId(rawValue) {
    const parsed = Number(rawValue);
    if (!Number.isInteger(parsed) || parsed <= 0) {
      return null;
    }
    return parsed;
  }

  function getTrackedRunIds() {
    if (mode === "detail") {
      const detailRunId = parseRunId(root.getAttribute("data-run-id") || "");
      return detailRunId === null ? [] : [detailRunId];
    }

    const rows = root.querySelectorAll("[data-run-id]");
    const runIds = [];
    for (const row of rows) {
      if (!(row instanceof HTMLElement)) {
        continue;
      }
      const runId = parseRunId(row.getAttribute("data-run-id") || "");
      if (runId === null) {
        continue;
      }
      runIds.push(runId);
    }
    return runIds;
  }

  function isEditableTextInput(element) {
    if (!(element instanceof HTMLInputElement)) {
      return false;
    }
    const inputType = (element.getAttribute("type") || "text").toLowerCase();
    return ["text", "number", "search", "url", "email", "tel", "password"].includes(inputType);
  }

  function isUserEditingFormControl() {
    const active = document.activeElement;
    if (!(active instanceof HTMLElement)) {
      return false;
    }
    if (active instanceof HTMLTextAreaElement || active instanceof HTMLSelectElement) {
      return true;
    }
    return isEditableTextInput(active);
  }

  function setStatusChipValue(chip, status) {
    chip.textContent = status;
    chip.className = "status-chip status-" + status;
  }

  function setElementTextById(id, value) {
    const element = document.getElementById(id);
    if (!(element instanceof HTMLElement)) {
      return;
    }
    element.textContent = value;
  }

  function updateDetailRunView(payload) {
    const run = payload.run && typeof payload.run === "object" ? payload.run : {};
    const statusValue = String(run.status || "");
    const statusChip = document.getElementById("suggestion-run-status-chip");
    if (statusChip instanceof HTMLElement) {
      setStatusChipValue(statusChip, statusValue);
    }

    setElementTextById("suggestion-run-target-count", String(run.target_count ?? 0));
    setElementTextById("suggestion-run-processed-count", String(run.processed_count ?? 0));
    setElementTextById("suggestion-run-success-count", String(run.success_count ?? 0));
    setElementTextById("suggestion-run-invalid-count", String(run.invalid_count ?? 0));
    setElementTextById("suggestion-run-error-count", String(run.error_count ?? 0));
    setElementTextById("suggestion-run-error", String(run.error || ""));
    setElementTextById("suggestion-run-started-at", String(run.started_at || ""));
    setElementTextById("suggestion-run-finished-at", String(run.finished_at || ""));

    const queueCounts = payload.queue_counts && typeof payload.queue_counts === "object" ? payload.queue_counts : {};
    const stageCountNodes = root.querySelectorAll("[data-stage-count]");
    for (const node of stageCountNodes) {
      if (!(node instanceof HTMLElement)) {
        continue;
      }
      const stageName = (node.getAttribute("data-stage-count") || "").trim();
      if (!stageName) {
        continue;
      }
      const rawCount = queueCounts[stageName];
      const normalizedCount = Number.isFinite(Number(rawCount)) ? String(Number(rawCount)) : "0";
      node.textContent = normalizedCount;
    }
  }

  function shouldHideRowForStatusFilter(statusFilter, runStatus) {
    if (statusFilter !== "queued" && statusFilter !== "running") {
      return false;
    }
    return runStatus !== statusFilter;
  }

  function updateListRunRow(runId, payload) {
    const row = root.querySelector('[data-run-id="' + String(runId) + '"]');
    if (!(row instanceof HTMLElement)) {
      return;
    }
    const run = payload.run && typeof payload.run === "object" ? payload.run : {};
    const statusValue = String(run.status || "");

    const statusChip = row.querySelector("[data-run-status]");
    if (statusChip instanceof HTMLElement) {
      setStatusChipValue(statusChip, statusValue);
    }

    const progressNode = row.querySelector("[data-run-progress]");
    if (progressNode instanceof HTMLElement) {
      progressNode.textContent = String(run.processed_count ?? 0) + " / " + String(run.target_count ?? 0);
    }

    const resultsNode = row.querySelector("[data-run-results]");
    if (resultsNode instanceof HTMLElement) {
      resultsNode.textContent = String(run.success_count ?? 0) + " ranked, " + String(run.invalid_count ?? 0) + " invalid";
    }

    const statusFilter = (root.getAttribute("data-status-filter") || "").trim().toLowerCase();
    row.hidden = shouldHideRowForStatusFilter(statusFilter, statusValue);
  }

  function recordRunState(runId, runPayload) {
    const nextStatus = String(runPayload.status || "");
    const nextUpdatedAt = typeof runPayload.updated_at === "string" ? runPayload.updated_at : null;
    const nextFinishedAt = typeof runPayload.finished_at === "string" ? runPayload.finished_at : null;
    const previousState = runStateById.get(runId) || { status: null, updatedAt: null, finishedAt: null };

    const hadPreviousStatus = typeof previousState.status === "string" && previousState.status.length > 0;
    const wasTerminal = typeof previousState.status === "string" && isTerminalStatus(previousState.status);
    const nowTerminal = isTerminalStatus(nextStatus);
    const completedNow = hadPreviousStatus && !wasTerminal && nowTerminal && Boolean(nextFinishedAt);

    runStateById.set(runId, {
      status: nextStatus,
      updatedAt: nextUpdatedAt,
      finishedAt: nextFinishedAt,
    });
    return {
      completedNow: completedNow,
      status: nextStatus,
    };
  }

  function requestReload() {
    if (reloadRequested) {
      return;
    }
    reloadRequested = true;
    window.location.reload();
  }

  function showDeferredReloadBanner() {
    hasPendingReload = true;
    if (refreshBanner instanceof HTMLElement) {
      refreshBanner.hidden = false;
    }
  }

  function handleCompletionDetected() {
    if (mode === "detail") {
      requestReload();
      return;
    }
    if (isUserEditingFormControl()) {
      showDeferredReloadBanner();
      return;
    }
    requestReload();
  }

  function shouldStopPolling(results) {
    if (!results.length) {
      return true;
    }
    let sawRunPayload = false;
    let allTerminal = true;
    for (const result of results) {
      if (result.type !== "ok") {
        continue;
      }
      sawRunPayload = true;
      if (!isTerminalStatus(result.status)) {
        allTerminal = false;
      }
    }
    if (!sawRunPayload) {
      return true;
    }
    return allTerminal && !hasPendingReload;
  }

  async function fetchRunStatus(runId) {
    const params = new URLSearchParams({
      profile_id: String(profileId),
      include_proposals: "false",
    });
    const response = await fetch("/api/suggestions/runs/" + String(runId) + "?" + params.toString(), {
      headers: { Accept: "application/json" },
    });

    if (response.status === 404) {
      return { type: "not_found", runId: runId };
    }
    if (!response.ok) {
      throw new Error("poll_failed_status_" + String(response.status));
    }

    const payload = await response.json();
    if (mode === "detail") {
      updateDetailRunView(payload);
    } else {
      updateListRunRow(runId, payload);
    }

    const runPayload = payload.run && typeof payload.run === "object" ? payload.run : {};
    const stateUpdate = recordRunState(runId, runPayload);
    return {
      type: "ok",
      runId: runId,
      status: stateUpdate.status,
      completedNow: stateUpdate.completedNow,
    };
  }

  async function runPollCycle() {
    if (stopped || reloadRequested) {
      return;
    }

    const runIds = getTrackedRunIds();
    if (!runIds.length) {
      stopped = true;
      clearTimer();
      return;
    }

    try {
      const settledResults = await Promise.allSettled(runIds.map((runId) => fetchRunStatus(runId)));
      const normalizedResults = [];
      let completionDetected = false;

      for (const settled of settledResults) {
        if (settled.status === "rejected") {
          throw settled.reason;
        }
        normalizedResults.push(settled.value);
        if (settled.value.type === "ok" && settled.value.completedNow) {
          completionDetected = true;
        }
      }

      failureCount = 0;
      if (completionDetected) {
        handleCompletionDetected();
        if (reloadRequested) {
          return;
        }
      }

      if (shouldStopPolling(normalizedResults)) {
        stopped = true;
        clearTimer();
        return;
      }

      schedule(activeSuccessDelayMs());
    } catch (_err) {
      const backoffIndex = Math.min(failureCount, FAILURE_INTERVALS_MS.length - 1);
      const delayMs = FAILURE_INTERVALS_MS[backoffIndex];
      failureCount += 1;
      schedule(delayMs);
    }
  }

  if (refreshNowButton instanceof HTMLButtonElement) {
    refreshNowButton.addEventListener("click", function () {
      requestReload();
    });
  }

  document.addEventListener("visibilitychange", function () {
    if (stopped || reloadRequested || failureCount > 0) {
      return;
    }
    schedule(activeSuccessDelayMs());
  });

  runPollCycle();
})();
