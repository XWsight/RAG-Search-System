"use strict";

const KEY_STORAGE = "rag-studio-api-key";
const ACTIVE_STORAGE = "rag-studio-active-base";

const state = {
  apiKey: readSession(KEY_STORAGE),
  knowledgeBases: [],
  activeId: readSession(ACTIVE_STORAGE),
  sessionId: `session_${randomId()}`,
  busy: false,
  search: "",
  pendingConsent: null,
};

const ui = {
  keyDialog: document.querySelector("#key-dialog"),
  keyForm: document.querySelector("#key-form"),
  keyInput: document.querySelector("#api-key"),
  keyError: document.querySelector("#key-error"),
  changeKey: document.querySelector("#change-key"),
  serviceDot: document.querySelector("#service-dot"),
  serviceLabel: document.querySelector("#service-label"),
  list: document.querySelector("#knowledge-list"),
  knowledgeSearch: document.querySelector("#knowledge-search"),
  mobileBaseSelect: document.querySelector("#mobile-base-select"),
  refresh: document.querySelector("#refresh-bases"),
  workspaceTitle: document.querySelector("#workspace-title"),
  emptyState: document.querySelector("#empty-state"),
  chatView: document.querySelector("#chat-view"),
  activeStatus: document.querySelector("#active-status"),
  activeStats: document.querySelector("#active-stats"),
  baseDetails: document.querySelector("#base-details"),
  taskBanner: document.querySelector("#task-banner"),
  taskTitle: document.querySelector("#task-title"),
  taskDetail: document.querySelector("#task-detail"),
  conversation: document.querySelector("#conversation"),
  openCreate: document.querySelector("#open-create"),
  emptyCreate: document.querySelector("#empty-create"),
  createDialog: document.querySelector("#create-dialog"),
  createForm: document.querySelector("#create-form"),
  closeCreate: document.querySelector("#close-create"),
  cancelCreate: document.querySelector("#cancel-create"),
  baseName: document.querySelector("#base-name"),
  documents: document.querySelector("#documents"),
  dropZone: document.querySelector("#drop-zone"),
  fileList: document.querySelector("#file-list"),
  createError: document.querySelector("#create-error"),
  submitCreate: document.querySelector("#submit-create"),
  deleteBase: document.querySelector("#delete-base"),
  clearChat: document.querySelector("#clear-chat"),
  questionForm: document.querySelector("#question-form"),
  question: document.querySelector("#question"),
  sendQuestion: document.querySelector("#send-question"),
  allowCloud: document.querySelector("#allow-cloud"),
  allowWeb: document.querySelector("#allow-web"),
  deepResearch: document.querySelector("#deep-research"),
  privacyNote: document.querySelector("#privacy-note"),
  detailsDialog: document.querySelector("#details-dialog"),
  closeDetails: document.querySelector("#close-details"),
  detailsTitle: document.querySelector("#details-title"),
  detailsMetrics: document.querySelector("#details-metrics"),
  documentList: document.querySelector("#document-list"),
  consentDialog: document.querySelector("#consent-dialog"),
  consentForm: document.querySelector("#consent-form"),
  consentTitle: document.querySelector("#consent-title"),
  consentDescription: document.querySelector("#consent-description"),
  consentBoundary: document.querySelector("#consent-boundary"),
  denyConsent: document.querySelector("#deny-consent"),
  grantConsent: document.querySelector("#grant-consent"),
  toastStack: document.querySelector("#toast-stack"),
};

function readSession(key) {
  try {
    return window.sessionStorage.getItem(key) || "";
  } catch {
    return "";
  }
}

function writeSession(key, value) {
  try {
    if (value) window.sessionStorage.setItem(key, value);
    else window.sessionStorage.removeItem(key);
  } catch {
    // The app remains usable when browser storage is unavailable.
  }
}

function randomId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID().replaceAll("-", "");
  const bytes = new Uint8Array(16);
  window.crypto.getRandomValues(bytes);
  return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = String(text);
  return element;
}

function formatBytes(value) {
  const size = Number(value) || 0;
  if (size < 1024) return `${size} B`;
  if (size < 1024 ** 2) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 ** 2).toFixed(1)} MB`;
}

function routeLabel(route) {
  return {
    local: "本地证据",
    web: "网络证据",
    hybrid: "混合证据",
    retrieval_only: "仅检索",
    refused: "证据不足",
    error: "服务降级",
  }[route] || route || "未知路径";
}

function statusLabel(status) {
  return {
    preparing: "准备文档",
    pending: "等待处理",
    indexing: "建立索引",
    ready: "可用",
    failed: "失败",
    cancelling: "正在取消",
    deleting: "正在删除",
  }[status] || status;
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.apiKey) headers.set("X-API-Key", state.apiKey);
  const response = await fetch(path, {...options, headers});
  let payload = null;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    payload = await response.json();
  }
  if (!response.ok) {
    const error = new Error(payload?.error?.message || `请求失败（${response.status}）`);
    error.code = payload?.error?.code || "request_failed";
    error.status = response.status;
    throw error;
  }
  return payload;
}

async function checkHealth() {
  try {
    const response = await fetch("/health/ready", {cache: "no-store"});
    if (!response.ok) throw new Error("not ready");
    ui.serviceDot.className = "status-dot online";
    ui.serviceLabel.textContent = "服务运行正常";
  } catch {
    ui.serviceDot.className = "status-dot offline";
    ui.serviceLabel.textContent = "服务暂不可用";
  }
}

function showKeyDialog(message = "") {
  ui.keyError.textContent = message;
  ui.keyInput.value = "";
  if (!ui.keyDialog.open) ui.keyDialog.showModal();
  window.setTimeout(() => ui.keyInput.focus(), 30);
}

async function connectWithKey(event) {
  event.preventDefault();
  const candidate = ui.keyInput.value.trim();
  if (candidate.length < 8) {
    ui.keyError.textContent = "请输入有效的访问密钥。";
    return;
  }
  const previous = state.apiKey;
  state.apiKey = candidate;
  ui.keyError.textContent = "正在验证……";
  try {
    await loadKnowledgeBases({selectDefault: true});
    writeSession(KEY_STORAGE, candidate);
    ui.keyDialog.close();
    toast("连接成功，密钥仅保存在当前标签页。", "success");
  } catch (error) {
    state.apiKey = previous;
    ui.keyError.textContent = error.status === 401 ? "密钥验证失败，请重新复制。" : error.message;
  }
}

function renderKnowledgeBases() {
  ui.list.replaceChildren();
  ui.mobileBaseSelect.replaceChildren();
  const mobilePlaceholder = node("option", "", state.knowledgeBases.length ? "切换知识库" : "暂无知识库");
  mobilePlaceholder.value = "";
  ui.mobileBaseSelect.append(mobilePlaceholder);
  for (const item of state.knowledgeBases) {
    const option = node("option", "", item.name);
    option.value = item.id;
    option.selected = item.id === state.activeId;
    ui.mobileBaseSelect.append(option);
  }
  if (!state.knowledgeBases.length) {
    ui.list.append(node("p", "list-empty", "还没有知识库。上传资料后会显示在这里。"));
    return;
  }
  const query = state.search.trim().toLocaleLowerCase("zh-CN");
  const visibleItems = state.knowledgeBases.filter((item) => item.name.toLocaleLowerCase("zh-CN").includes(query));
  if (!visibleItems.length) {
    ui.list.append(node("p", "list-empty", "没有匹配的知识库。"));
    return;
  }
  for (const item of visibleItems) {
    const button = node("button", `knowledge-item${item.id === state.activeId ? " active" : ""}`);
    button.type = "button";
    button.dataset.id = item.id;
    button.setAttribute("aria-label", `打开知识库 ${item.name}`);
    const icon = node("span", "knowledge-icon", item.name.trim().slice(0, 1).toUpperCase() || "K");
    const copy = node("span", "knowledge-copy");
    copy.append(node("span", "knowledge-name", item.name));
    const detail = node("span", "knowledge-detail");
    detail.append(node("span", `mini-status ${item.status}`));
    detail.append(node("span", "", `${statusLabel(item.status)} · ${item.document_count} 份文档`));
    copy.append(detail);
    button.append(icon, copy);
    button.addEventListener("click", () => selectKnowledgeBase(item.id));
    ui.list.append(button);
  }
}

async function loadKnowledgeBases({selectDefault = false} = {}) {
  const payload = await api("/v1/knowledge-bases?limit=100&offset=0");
  state.knowledgeBases = payload.items || [];
  if (state.activeId && !state.knowledgeBases.some((item) => item.id === state.activeId)) {
    state.activeId = "";
  }
  if (!state.activeId && selectDefault) {
    state.activeId = state.knowledgeBases.find((item) => item.status === "ready")?.id || "";
  }
  writeSession(ACTIVE_STORAGE, state.activeId);
  renderKnowledgeBases();
  renderWorkspace();
}

function activeKnowledgeBase() {
  return state.knowledgeBases.find((item) => item.id === state.activeId) || null;
}

function selectKnowledgeBase(id) {
  state.activeId = id;
  state.sessionId = `session_${randomId()}`;
  writeSession(ACTIVE_STORAGE, id);
  renderKnowledgeBases();
  resetConversation();
  renderWorkspace();
}

function renderWorkspace() {
  const active = activeKnowledgeBase();
  const hasActive = Boolean(active);
  ui.emptyState.classList.toggle("hidden", hasActive);
  ui.chatView.classList.toggle("hidden", !hasActive);
  ui.baseDetails.classList.toggle("hidden", !hasActive);
  if (!active) {
    ui.workspaceTitle.textContent = state.knowledgeBases.length ? "选择一个知识库" : "你的知识工作台";
    return;
  }
  ui.workspaceTitle.textContent = active.name;
  ui.activeStatus.textContent = String(active.status).toUpperCase();
  ui.activeStatus.className = `ready-pill ${active.status}`;
  ui.activeStats.textContent = `${active.document_count} 份文档 · ${active.chunk_count} 个片段 · ${formatBytes(active.total_bytes)}`;
  const ready = active.status === "ready";
  ui.question.disabled = !ready;
  ui.sendQuestion.disabled = !ready;
  ui.question.placeholder = ready ? "基于当前知识库提问……" : `知识库${statusLabel(active.status)}，完成后即可提问`;
}

function openCreateDialog() {
  ui.createError.textContent = "";
  ui.createForm.reset();
  renderFiles();
  ui.createDialog.showModal();
  window.setTimeout(() => ui.baseName.focus(), 30);
}

function closeCreateDialog() {
  if (ui.createDialog.open) ui.createDialog.close();
}

function renderFiles() {
  ui.fileList.replaceChildren();
  Array.from(ui.documents.files || []).forEach((file, index) => {
    const item = node("div", "file-entry");
    item.append(node("span", "file-entry-name", file.name));
    const meta = node("span", "file-entry-meta");
    meta.append(node("span", "", formatBytes(file.size)));
    const remove = node("button", "remove-file", "×");
    remove.type = "button";
    remove.setAttribute("aria-label", `移除 ${file.name}`);
    remove.addEventListener("click", () => removeSelectedFile(index));
    meta.append(remove);
    item.append(meta);
    ui.fileList.append(item);
  });
}

function removeSelectedFile(index) {
  const transfer = new DataTransfer();
  Array.from(ui.documents.files || []).forEach((file, position) => {
    if (position !== index) transfer.items.add(file);
  });
  ui.documents.files = transfer.files;
  renderFiles();
}

function installDroppedFiles(files) {
  const transfer = new DataTransfer();
  Array.from(files).slice(0, 10).forEach((file) => transfer.items.add(file));
  ui.documents.files = transfer.files;
  renderFiles();
}

async function createKnowledgeBase(event) {
  event.preventDefault();
  const name = ui.baseName.value.trim();
  const files = Array.from(ui.documents.files || []);
  if (!name || !files.length) {
    ui.createError.textContent = "请填写名称并至少选择一个文档。";
    return;
  }
  if (files.length > 10) {
    ui.createError.textContent = "一次最多上传 10 个文档。";
    return;
  }
  ui.submitCreate.disabled = true;
  ui.createError.textContent = "正在安全上传……";
  showTask("正在上传文档", `${files.length} 个文件正在写入受保护的知识库空间`);
  const form = new FormData();
  form.append("name", name);
  files.forEach((file) => form.append("files", file, file.name));
  try {
    const payload = await api("/v1/knowledge-bases", {
      method: "POST",
      headers: {"Idempotency-Key": `create-${Date.now()}-${randomId().slice(0, 16)}`},
      body: form,
    });
    closeCreateDialog();
    state.activeId = payload.knowledge_base.id;
    writeSession(ACTIVE_STORAGE, state.activeId);
    toast("文档已接收，正在建立混合检索索引。", "success");
    showTask("正在建立混合索引", "执行文档解析、分块、向量化和 BM25 构建");
    await loadKnowledgeBases();
    await pollJob(payload.job_id, payload.knowledge_base.id);
  } catch (error) {
    hideTask();
    ui.createError.textContent = error.message;
    if (error.status === 401) showKeyDialog("访问密钥已失效，请重新连接。");
  } finally {
    ui.submitCreate.disabled = false;
  }
}

async function pollJob(jobId, knowledgeBaseId) {
  for (let attempt = 0; attempt < 150; attempt += 1) {
    await delay(1200);
    try {
      const job = await api(`/v1/jobs/${encodeURIComponent(jobId)}`);
      if (attempt % 2 === 0 || ["succeeded", "failed", "cancelled"].includes(job.status)) {
        await loadKnowledgeBases();
      }
      if (job.status === "succeeded") {
        state.activeId = knowledgeBaseId;
        writeSession(ACTIVE_STORAGE, knowledgeBaseId);
        renderKnowledgeBases();
        renderWorkspace();
        toast("知识库已就绪，可以开始提问。", "success");
        showTask("知识库已就绪", "索引构建完成，正在进入问答工作区", {complete: true});
        window.setTimeout(hideTask, 2600);
        return;
      }
      if (["failed", "cancelled"].includes(job.status)) {
        hideTask();
        toast("索引任务未完成，请检查文档格式后重试。", "error");
        return;
      }
    } catch (error) {
      if (error.status === 429) await delay(1600);
      else if (error.status === 404) {
        hideTask();
        return;
      }
      else toast(error.message, "error");
    }
  }
  toast("索引仍在后台运行，请稍后刷新状态。", "error");
  hideTask();
}

function showTask(title, detail, {complete = false} = {}) {
  ui.taskTitle.textContent = title;
  ui.taskDetail.textContent = detail;
  ui.taskBanner.classList.remove("hidden");
  ui.taskBanner.classList.toggle("complete", complete);
  const spinner = ui.taskBanner.querySelector(".task-spinner");
  if (spinner) spinner.hidden = complete;
}

function hideTask() {
  ui.taskBanner.classList.add("hidden");
  ui.taskBanner.classList.remove("complete");
  const spinner = ui.taskBanner.querySelector(".task-spinner");
  if (spinner) spinner.hidden = false;
}

function addMessage(role, content, options = {}) {
  const article = node("article", `message ${role}${options.pending ? " pending" : ""}`);
  article.append(node("div", "message-label", role === "user" ? "你" : "RAG Studio"));
  const body = node("div", "message-body");
  if (options.pending) {
    const dots = node("span", "typing-dots");
    dots.append(node("span"), node("span"), node("span"));
    body.append(dots);
  } else {
    body.textContent = content;
  }
  article.append(body);
  if (options.result) appendAnswerDetails(article, options.result);
  ui.conversation.append(article);
  ui.conversation.scrollTo({top: ui.conversation.scrollHeight, behavior: "smooth"});
  return article;
}

function appendAnswerDetails(article, result) {
  const meta = node("div", "answer-meta");
  meta.append(node("span", "", routeLabel(result.decision?.route)));
  if (typeof result.decision?.confidence === "number") {
    meta.append(node("span", "", `置信度 ${Math.round(result.decision.confidence * 100)}%`));
  }
  if (typeof result.latency_ms === "number") {
    meta.append(node("span", "", `${Math.round(result.latency_ms)} ms`));
  }
  if (Array.isArray(result.claims) && result.claims.length) {
    meta.append(node("span", "", `${result.claims.length} 条逐项引用`));
  }
  const copy = node("button", "copy-answer", "复制回答");
  copy.type = "button";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(result.answer || "");
      copy.textContent = "已复制";
      window.setTimeout(() => { copy.textContent = "复制回答"; }, 1500);
    } catch {
      toast("浏览器没有授予剪贴板权限。", "error");
    }
  });
  meta.append(copy);
  article.append(meta);
  const citations = node("div", "citations");
  for (const citation of result.citations || []) {
    const details = node("details", "citation");
    const score = typeof citation.score === "number" ? ` · ${Math.round(citation.score * 100)}%` : "";
    details.append(node("summary", "", `[${citation.id}] ${citation.source_name}${score}`));
    details.append(node("p", "", citation.excerpt));
    if (citation.url) {
      const link = node("a", "", "打开原始来源 ↗");
      link.href = citation.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      details.append(link);
    }
    citations.append(details);
  }
  if (citations.childElementCount) article.append(citations);
}

async function askQuestion(event) {
  event.preventDefault();
  const active = activeKnowledgeBase();
  const question = ui.question.value.trim();
  if (!active || active.status !== "ready" || !question || state.busy) return;
  state.busy = true;
  ui.sendQuestion.disabled = true;
  ui.question.value = "";
  resizeQuestion();
  addMessage("user", question);
  const pending = addMessage("assistant", "", {pending: true});
  try {
    const result = await api("/v1/answers", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        knowledge_base_id: active.id,
        question,
        session_id: state.sessionId,
        allow_cloud: ui.allowCloud.checked,
        allow_web: ui.allowWeb.checked,
        deep_research: ui.deepResearch.checked,
      }),
    });
    pending.remove();
    addMessage("assistant", result.answer, {result});
  } catch (error) {
    pending.remove();
    addMessage("assistant", `处理失败：${error.message}`);
    if (error.status === 401) showKeyDialog("访问密钥已失效，请重新连接。");
  } finally {
    state.busy = false;
    ui.sendQuestion.disabled = false;
    ui.question.focus();
  }
}

async function clearConversation() {
  const active = activeKnowledgeBase();
  if (active) {
    try {
      await api(`/v1/knowledge-bases/${encodeURIComponent(active.id)}/sessions/${encodeURIComponent(state.sessionId)}`, {method: "DELETE"});
    } catch (error) {
      if (error.status !== 404) toast(error.message, "error");
    }
  }
  state.sessionId = `session_${randomId()}`;
  resetConversation();
  toast("当前对话记忆已清空。", "success");
}

function resetConversation() {
  ui.conversation.replaceChildren();
  const card = node("article", "welcome-card");
  card.append(node("p", "eyebrow", "知识库已就绪"));
  card.append(node("h2", "", "现在可以围绕资料提问"));
  card.append(node("p", "", "回答会展示检索路径、置信度和引用片段。开启云端或联网前，界面会明确显示数据边界。"));
  const suggestions = node("div", "suggestion-list");
  [
    "概括这份知识库的核心内容",
    "列出资料中最重要的三个观点",
    "哪些结论需要进一步核验？",
  ].forEach((question) => {
    const button = node("button", "suggestion-button", question);
    button.type = "button";
    button.addEventListener("click", () => {
      ui.question.value = question;
      resizeQuestion();
      ui.question.focus();
    });
    suggestions.append(button);
  });
  card.append(suggestions);
  ui.conversation.append(card);
}

async function deleteActiveKnowledgeBase() {
  const active = activeKnowledgeBase();
  if (!active) return;
  const confirmed = window.confirm(`确定删除“${active.name}”吗？原始文档、索引和对话记忆将被永久移除。`);
  if (!confirmed) return;
  ui.deleteBase.disabled = true;
  try {
    await api(`/v1/knowledge-bases/${encodeURIComponent(active.id)}`, {method: "DELETE"});
    state.activeId = "";
    writeSession(ACTIVE_STORAGE, "");
    resetConversation();
    await loadKnowledgeBases({selectDefault: true});
    toast("知识库及其持久化数据已删除。", "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    ui.deleteBase.disabled = false;
  }
}

function openDetailsDialog() {
  const active = activeKnowledgeBase();
  if (!active) return;
  ui.detailsTitle.textContent = active.name;
  ui.detailsMetrics.replaceChildren();
  [
    [active.document_count, "文档数量"],
    [active.chunk_count, "检索片段"],
    [formatBytes(active.total_bytes), "原始资料"],
  ].forEach(([value, label]) => {
    const metric = node("div", "detail-metric");
    metric.append(node("strong", "", value), node("span", "", label));
    ui.detailsMetrics.append(metric);
  });
  ui.documentList.replaceChildren();
  for (const documentInfo of active.documents || []) {
    const entry = node("div", "document-entry");
    const suffix = documentInfo.name.includes(".") ? documentInfo.name.split(".").pop().slice(0, 5).toUpperCase() : "DOC";
    entry.append(node("span", "document-type", suffix));
    const copy = node("span", "");
    copy.append(node("span", "document-name", documentInfo.name));
    copy.append(node("span", "document-meta", `${formatBytes(documentInfo.size_bytes)} · SHA-256 ${documentInfo.sha256.slice(0, 12)}…`));
    entry.append(copy);
    ui.documentList.append(entry);
  }
  if (!ui.documentList.childElementCount) {
    ui.documentList.append(node("p", "list-empty", "该知识库暂时没有可展示的文档清单。"));
  }
  ui.detailsDialog.showModal();
}

function closeDetailsDialog() {
  if (ui.detailsDialog.open) ui.detailsDialog.close();
}

function requestExternalConsent(control) {
  if (!control.checked) {
    updatePrivacyNote();
    return;
  }
  control.checked = false;
  state.pendingConsent = control;
  const cloud = control === ui.allowCloud;
  ui.consentTitle.textContent = cloud ? "开启云端生成？" : "开启联网补充？";
  ui.consentDescription.textContent = cloud
    ? "系统会把当前问题和命中的检索证据发送给已配置的智谱模型，用于组织自然语言回答。"
    : "当本地证据不足时，系统会把当前问题发送给已配置的网络搜索服务，并展示外部来源。";
  ui.consentBoundary.textContent = cloud
    ? "发送：问题、必要的对话上下文、检索证据。不发送：本地访问密钥、租户标识、未命中的完整文档。"
    : "发送：当前问题或有界查询计划。不发送：本地访问密钥、完整知识库和未命中文档。";
  ui.consentDialog.showModal();
}

function grantExternalConsent(event) {
  event.preventDefault();
  if (state.pendingConsent) state.pendingConsent.checked = true;
  state.pendingConsent = null;
  ui.consentDialog.close();
  updatePrivacyNote();
  toast("外部服务仅对当前请求开关生效，你可以随时关闭。", "success");
}

function denyExternalConsent() {
  if (state.pendingConsent) state.pendingConsent.checked = false;
  state.pendingConsent = null;
  ui.consentDialog.close();
  updatePrivacyNote();
}

function updatePrivacyNote() {
  const destinations = [];
  if (ui.allowCloud.checked) destinations.push("云端模型");
  if (ui.allowWeb.checked) destinations.push("网络搜索");
  if (destinations.length) {
    ui.privacyNote.textContent = `问题${ui.allowCloud.checked ? "与检索证据" : ""}将发送到${destinations.join("、")}`;
    ui.privacyNote.classList.add("external");
  } else {
    ui.privacyNote.textContent = "内容不会发送到外部服务";
    ui.privacyNote.classList.remove("external");
  }
}

function resizeQuestion() {
  ui.question.style.height = "auto";
  ui.question.style.height = `${Math.min(ui.question.scrollHeight, 160)}px`;
}

function toast(message, kind = "success") {
  const item = node("div", `toast ${kind}`, message);
  ui.toastStack.append(item);
  window.setTimeout(() => item.remove(), 4200);
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function bindEvents() {
  ui.keyForm.addEventListener("submit", connectWithKey);
  ui.changeKey.addEventListener("click", () => {
    state.apiKey = "";
    writeSession(KEY_STORAGE, "");
    showKeyDialog();
  });
  ui.knowledgeSearch.addEventListener("input", () => {
    state.search = ui.knowledgeSearch.value;
    renderKnowledgeBases();
  });
  ui.mobileBaseSelect.addEventListener("change", () => {
    if (ui.mobileBaseSelect.value) selectKnowledgeBase(ui.mobileBaseSelect.value);
  });
  ui.refresh.addEventListener("click", () => loadKnowledgeBases().catch((error) => toast(error.message, "error")));
  ui.openCreate.addEventListener("click", openCreateDialog);
  ui.emptyCreate.addEventListener("click", openCreateDialog);
  ui.closeCreate.addEventListener("click", closeCreateDialog);
  ui.cancelCreate.addEventListener("click", closeCreateDialog);
  ui.createForm.addEventListener("submit", createKnowledgeBase);
  ui.documents.addEventListener("change", renderFiles);
  ui.dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    ui.dropZone.classList.add("dragging");
  });
  ui.dropZone.addEventListener("dragleave", () => ui.dropZone.classList.remove("dragging"));
  ui.dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    ui.dropZone.classList.remove("dragging");
    installDroppedFiles(event.dataTransfer.files);
  });
  ui.createDialog.addEventListener("click", (event) => {
    if (event.target === ui.createDialog) closeCreateDialog();
  });
  ui.questionForm.addEventListener("submit", askQuestion);
  ui.question.addEventListener("input", resizeQuestion);
  ui.question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      ui.questionForm.requestSubmit();
    }
  });
  ui.clearChat.addEventListener("click", clearConversation);
  ui.deleteBase.addEventListener("click", deleteActiveKnowledgeBase);
  ui.baseDetails.addEventListener("click", openDetailsDialog);
  ui.closeDetails.addEventListener("click", closeDetailsDialog);
  ui.detailsDialog.addEventListener("click", (event) => {
    if (event.target === ui.detailsDialog) closeDetailsDialog();
  });
  ui.allowCloud.addEventListener("change", () => requestExternalConsent(ui.allowCloud));
  ui.allowWeb.addEventListener("change", () => requestExternalConsent(ui.allowWeb));
  ui.consentForm.addEventListener("submit", grantExternalConsent);
  ui.denyConsent.addEventListener("click", denyExternalConsent);
  ui.consentDialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    denyExternalConsent();
  });
  ui.deepResearch.addEventListener("change", () => {
    if (ui.deepResearch.checked && !ui.allowCloud.checked) {
      toast("深度研究可在纯本地模式运行；开启云端后还会使用查询规划。", "success");
    }
  });
}

async function start() {
  bindEvents();
  updatePrivacyNote();
  resetConversation();
  await checkHealth();
  if (!state.apiKey) {
    showKeyDialog();
    return;
  }
  try {
    await loadKnowledgeBases({selectDefault: true});
  } catch (error) {
    if (error.status === 401) showKeyDialog("当前密钥无效，请重新连接。");
    else toast(error.message, "error");
  }
}

start();
