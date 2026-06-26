const state = {
  config: null,
  runId: null,
  runStatus: null,
  pollTimer: null,
  selectedDepth: 5,
  selectedProvider: "codex_cli",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const FIXED_AGENTS = [
  "Bull/Bear Advocates",
  "Research Evaluator",
  "Trader",
  "Risk Analysts",
  "Portfolio Manager",
];

function formatDuration(seconds) {
  const mins = Math.floor(seconds / 60).toString().padStart(2, "0");
  const secs = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${mins}:${secs}`;
}

function selectedAnalysts() {
  return $$(".toggle.active").map((button) => button.dataset.analyst);
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdownish(markdown) {
  let html = escapeHtml(markdown);
  html = html.replace(/^### (.*)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.*)$/gm, "<h2>$1</h2>");
  html = html.replace(/^# (.*)$/gm, "<h1>$1</h1>");
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/^- (.*)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>");
  html = html
    .split(/\n{2,}/)
    .map((block) => {
      if (block.startsWith("<h") || block.startsWith("<ul")) return block;
      return `<p>${block.replace(/\n/g, "<br>")}</p>`;
    })
    .join("");
  return html;
}

function currentDepthLabel() {
  return state.selectedDepth === 1 ? "Shallow" : state.selectedDepth === 3 ? "Medium" : "Deep";
}

function currentEffort() {
  return state.config.depth_efforts[String(state.selectedDepth)] || "provider default";
}

function hasActiveRun() {
  return ["queued", "running"].includes(state.runStatus);
}

function syncStartButton() {
  $("#ticker-form button").disabled = hasActiveRun();
}

function plannedAgents() {
  const analysts = selectedAnalysts().map((key) => ({
    name: state.config.analysts[key] || key,
    status: "pending",
  }));
  return analysts.concat(FIXED_AGENTS.map((name) => ({ name, status: "pending" })));
}

function renderPlannedConfig() {
  const ticker = $("#ticker-input").value.trim().toUpperCase() || "Not selected";
  $("#config-summary").innerHTML = `
    <h2>Analysis Configuration</h2>
    <ul class="config-list">
      <li><strong>Ticker:</strong> ${escapeHtml(ticker)}</li>
      <li><strong>Analysis Date:</strong> ${escapeHtml($("#analysis-date").value || state.config.default_date)}</li>
      <li><strong>Analyst Team:</strong> ${selectedAnalysts().map((key) => escapeHtml(state.config.analysts[key] || key)).join(", ")}</li>
      <li><strong>Research Depth:</strong> ${escapeHtml(currentDepthLabel())}</li>
      <li><strong>Local CLI Effort:</strong> ${escapeHtml(currentEffort())}</li>
      <li><strong>Quick Think Model:</strong> ${escapeHtml(state.selectedProvider)}/${escapeHtml($("#quick-model").value || state.config.quick_model)}</li>
      <li><strong>Deep Think Model:</strong> ${escapeHtml(state.selectedProvider)}/${escapeHtml($("#deep-model").value || state.config.deep_model)}</li>
    </ul>`;
}

function renderReadyState() {
  if (!state.config || state.runId) return;
  state.runStatus = null;
  $("#progress-title").textContent = "Ready";
  $("#duration").textContent = "No active run";
  $("#stage-status span:last-child").textContent = "configure analysis";
  $("#stage-status .spinner").style.display = "none";
  $("#stop-run").disabled = true;
  $("#report-view").classList.add("hidden");
  $("#report-view").innerHTML = "";

  const agents = plannedAgents();
  const grouped = groupAgents(agents);
  renderAgentList($("#analyst-agents"), grouped.analyst);
  renderAgentList($("#research-agents"), grouped.research);
  renderAgentList($("#trader-agents"), grouped.trader);
  renderAgentList($("#risk-agents"), grouped.risk);
  renderAgentList($("#portfolio-agents"), grouped.portfolio);
  renderPipeline(agents);
  renderPlannedConfig();
  renderActivity({ stats: {}, activity: [] });
  syncStartButton();
}

function resetWorkspace() {
  history.pushState(null, "", "/");
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
  state.runId = null;
  state.runStatus = null;
  $("#ticker-input").value = "";
  renderReadyState();
}

async function loadConfig() {
  const response = await fetch("/api/config");
  state.config = await response.json();
  state.selectedProvider = state.config.provider_defaults[state.config.llm_provider]
    ? state.config.llm_provider
    : "codex_cli";
  $("#analysis-date").value = state.config.default_date;
  applyProviderDefaults();
  syncProviderControls();
}

function applyProviderDefaults() {
  const defaults = state.config.provider_defaults[state.selectedProvider] || {
    quick_model: state.config.quick_model,
    deep_model: state.config.deep_model,
  };
  $("#quick-model").value = defaults.quick_model;
  $("#deep-model").value = defaults.deep_model;
}

function syncProviderControls() {
  $$(".provider-segment button").forEach((button) => {
    button.classList.toggle("active", button.dataset.provider === state.selectedProvider);
  });
}

function buildRunRequest(ticker) {
  return {
    ticker,
    analysis_date: $("#analysis-date").value || state.config.default_date,
    analysts: selectedAnalysts(),
    research_depth: state.selectedDepth,
    llm_provider: state.selectedProvider,
    quick_model: $("#quick-model").value || state.config.quick_model,
    deep_model: $("#deep-model").value || state.config.deep_model,
    asset_type: ticker.toUpperCase().includes("-") ? "crypto" : "stock",
    output_language: "English",
  };
}

async function startRun(ticker) {
  if (hasActiveRun()) return;
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
  state.runStatus = "queued";
  syncStartButton();
  const response = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildRunRequest(ticker)),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Run failed to start (${response.status})`);
  }
  const payload = await response.json();
  state.runId = payload.run_id;
  history.pushState(null, "", `/r/${state.runId}`);
  $("#stage-status span:last-child").textContent = "starting analysis...";
  $("#stage-status .spinner").style.display = "inline-block";
  $("#stop-run").disabled = false;
  await pollRun();
  state.pollTimer = setInterval(pollRun, 1500);
}

async function pollRun() {
  if (!state.runId) return;
  const response = await fetch(`/api/runs/${state.runId}`);
  if (!response.ok) return;
  const run = await response.json();
  renderRun(run);
  if (["completed", "error", "cancelled"].includes(run.status) && state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function cancelRun() {
  if (!state.runId) return;
  $("#stop-run").disabled = true;
  await fetch(`/api/runs/${state.runId}/cancel`, { method: "POST" }).catch(() => {});
  await pollRun();
}

async function newAnalysis() {
  if (hasActiveRun()) {
    await cancelRun();
  }
  resetWorkspace();
}

function groupAgents(agents) {
  const names = new Map(agents.map((agent) => [agent.name, agent]));
  return {
    analyst: ["Market Analyst", "Social Media Analyst", "News Analyst", "Fundamentals Analyst"].map((name) => names.get(name)).filter(Boolean),
    research: ["Bull/Bear Advocates", "Research Evaluator"].map((name) => names.get(name)).filter(Boolean),
    trader: ["Trader"].map((name) => names.get(name)).filter(Boolean),
    risk: ["Risk Analysts"].map((name) => names.get(name)).filter(Boolean),
    portfolio: ["Portfolio Manager"].map((name) => names.get(name)).filter(Boolean),
  };
}

function renderAgentList(target, agents) {
  target.innerHTML = agents
    .map((agent) => {
      const status = agent.status === "completed" ? "completed" : agent.status === "running" ? "researching" : "pending";
      return `
        <div class="agent-row ${escapeHtml(agent.status)}">
          <span>${escapeHtml(agent.name)}</span>
          <span class="status-pill">${escapeHtml(status)}</span>
        </div>`;
    })
    .join("");
}

function renderPipeline(agents) {
  $("#pipeline").innerHTML = agents
    .map((agent) => `
      <div class="pipeline-row ${escapeHtml(agent.status)}">
        <span class="node-dot"></span>
        <span>${escapeHtml(agent.name)}</span>
        <span>${agent.status === "completed" ? "done" : agent.status}</span>
      </div>`)
    .join("");
}

function renderConfig(run) {
  $("#config-summary").innerHTML = `
    <h2>Analysis Configuration</h2>
    <ul class="config-list">
      <li><strong>Ticker:</strong> ${escapeHtml(run.request.ticker)}</li>
      <li><strong>Analysis Date:</strong> ${escapeHtml(run.request.analysis_date)}</li>
      <li><strong>Analyst Team:</strong> ${run.request.analysts.map((key) => escapeHtml(state.config.analysts[key] || key)).join(", ")}</li>
      <li><strong>Research Depth:</strong> ${escapeHtml(run.config.research_depth_label)}</li>
      <li><strong>Local CLI Effort:</strong> ${escapeHtml(run.config.local_cli_effort || "provider default")}</li>
      <li><strong>Quick Think Model:</strong> ${escapeHtml(run.config.llm_provider)}/${escapeHtml(run.config.quick_model)}</li>
      <li><strong>Deep Think Model:</strong> ${escapeHtml(run.config.llm_provider)}/${escapeHtml(run.config.deep_model)}</li>
    </ul>`;
}

function renderReport(run) {
  const key = run.current_report_key;
  if (!key || !run.reports[key]) {
    $("#report-view").classList.add("hidden");
    return;
  }

  const title = run.report_titles[key] || "Report";
  const final = key === "final_trade_decision";
  $("#report-view").classList.remove("hidden");
  $("#report-view").innerHTML = `
    <h1>${escapeHtml(title)}</h1>
    ${final ? `<div class="report-card"><strong>Final Recommendation</strong><br>${escapeHtml(run.signal || "Review")}</div>` : ""}
    <div>${renderMarkdownish(run.reports[key])}</div>
    ${run.report_path ? `<p class="saved-path">Saved report: ${escapeHtml(run.report_path)}</p>` : ""}`;
}

function renderActivity(run) {
  const stats = run.stats || {};
  $("#stats").textContent = `${stats.llm_calls || 0} LLM calls · ${stats.tool_calls || 0} tool calls · ${stats.tokens_in || 0} in, ${stats.tokens_out || 0} out`;
  $("#activity-log").innerHTML = (run.activity || [])
    .slice()
    .reverse()
    .map((item) => `
      <div class="activity-item">
        <time>${escapeHtml(item.time)}</time>
        <strong>${escapeHtml(item.agent)}</strong>
        <span>${escapeHtml(item.message)}</span>
      </div>`)
    .join("") || `<div class="activity-item">Activity log will appear as analysis runs.</div>`;
}

function renderRun(run) {
  state.runStatus = run.status;
  $("#ticker-input").value = run.request.ticker;
  $("#progress-title").textContent = `Analyzing ${run.request.ticker}`;
  $("#duration").textContent = `Duration: ${formatDuration(run.elapsed_seconds)}`;
  const statusText = run.status === "completed"
    ? "analysis complete"
    : run.status === "error"
      ? "analysis failed"
      : run.status === "cancelled"
        ? "analysis cancelled"
        : "preparing analysis...";
  $("#stage-status span:last-child").textContent = statusText;
  $("#stage-status .spinner").style.display = ["completed", "error", "cancelled"].includes(run.status) ? "none" : "inline-block";
  $("#stop-run").disabled = ["completed", "error", "cancelled"].includes(run.status);
  syncStartButton();

  const grouped = groupAgents(run.agents);
  renderAgentList($("#analyst-agents"), grouped.analyst);
  renderAgentList($("#research-agents"), grouped.research);
  renderAgentList($("#trader-agents"), grouped.trader);
  renderAgentList($("#risk-agents"), grouped.risk);
  renderAgentList($("#portfolio-agents"), grouped.portfolio);
  renderPipeline(run.agents);
  renderConfig(run);
  renderReport(run);
  renderActivity(run);

  if (run.status === "error") {
    $("#report-view").classList.remove("hidden");
    $("#report-view").innerHTML = `<h1>Analysis Failed</h1><div class="report-card">${escapeHtml(run.error)}</div>`;
  }

  if (run.status === "cancelled") {
    $("#report-view").classList.remove("hidden");
    $("#report-view").innerHTML = `<h1>Analysis Cancelled</h1><div class="report-card">The local analysis run was stopped.</div>`;
  }
}

function wireControls() {
  $("#ticker-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const ticker = $("#ticker-input").value.trim().toUpperCase();
    if (!ticker) return;
    $("#ticker-form button").disabled = true;
    try {
      await startRun(ticker);
    } catch (error) {
      state.runStatus = null;
      alert(error.message);
    } finally {
      syncStartButton();
    }
  });

  $("#new-run").addEventListener("click", newAnalysis);
  $("#stop-run").addEventListener("click", cancelRun);
  $("#ticker-input").addEventListener("input", renderReadyState);
  $("#analysis-date").addEventListener("input", renderReadyState);
  $("#quick-model").addEventListener("input", renderReadyState);
  $("#deep-model").addEventListener("input", renderReadyState);

  $$(".toggle").forEach((button) => {
    button.addEventListener("click", () => {
      button.classList.toggle("active");
      if (selectedAnalysts().length === 0) button.classList.add("active");
      renderReadyState();
    });
  });

  $$(".depth-segment button").forEach((button) => {
    button.addEventListener("click", () => {
      $$(".depth-segment button").forEach((candidate) => candidate.classList.remove("active"));
      button.classList.add("active");
      state.selectedDepth = Number(button.dataset.depth);
      renderReadyState();
    });
  });

  $$(".provider-segment button").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedProvider = button.dataset.provider;
      applyProviderDefaults();
      syncProviderControls();
      renderReadyState();
    });
  });
}

async function boot() {
  await loadConfig();
  wireControls();
  const match = location.pathname.match(/^\/r\/([a-f0-9]+)/);
  if (match) {
    state.runId = match[1];
    await pollRun();
    state.pollTimer = setInterval(pollRun, 1500);
  } else {
    resetWorkspace();
  }
}

boot().catch((error) => {
  console.error(error);
  alert(`Local Trading Desk UI failed to load: ${error.message}`);
});
