const $ = (s) => document.querySelector(s);
const state = { sources: [], source: "", entries: [], stats: null, filter: "readable", expanded: new Set(["viking", "viking/team-main", "viking/team-main/user"]), selected: "", timer: null };
const icons = { markdown: "M↓", json: "{ }", text: "T", sqlite: "DB", binary: "01", empty: "·" };
const labels = { markdown: "Markdown", json: "JSON", text: "文本", sqlite: "SQLite", binary: "二进制", empty: "空文件" };

function esc(value = "") {
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}
function formatBytes(n) {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.min(Math.floor(Math.log(n) / Math.log(1024)), 3);
  return (n / 1024 ** i).toFixed(i ? 1 : 0) + " " + units[i];
}
function toast(message) {
  const node = $("#toast"); node.textContent = message; node.classList.add("show");
  clearTimeout(toast.timer); toast.timer = setTimeout(() => node.classList.remove("show"), 1800);
}
async function api(url) {
  const response = await fetch(url); const data = await response.json();
  if (!response.ok) { const error = new Error(data.error || "HTTP " + response.status); error.data = data; throw error; }
  return data;
}
function sourceQuery(extra = {}) { return new URLSearchParams({ source: state.source, ...extra }); }
async function loadSources() {
  const data = await api("/api/sources"); state.sources = data.sources;
  const remembered = localStorage.getItem("data-reader-source");
  state.source = data.sources.some((source) => source.id === remembered) ? remembered : data.sources[0].id;
  $("#source-select").innerHTML = data.sources.map((source) => '<option value="' + esc(source.id) + '">' + esc(source.label) + "</option>").join("");
  $("#source-select").value = state.source; renderSourceDescription();
}
function renderSourceDescription() {
  const source = state.sources.find((item) => item.id === state.source);
  $("#source-description").textContent = source ? source.description : "";
}
async function loadIndex() {
  $("#tree").innerHTML = '<div class="empty-state">正在建立索引…</div>';
  const data = await api("/api/index?" + sourceQuery()); state.entries = data.entries; state.stats = data.stats;
  renderStats(); renderKinds(); renderTree();
}
function renderStats() {
  $("#stats").innerHTML = [[state.stats.files.toLocaleString(), "文件"], [state.stats.readable.toLocaleString(), "可阅读"], [formatBytes(state.stats.bytes), "磁盘内容"]]
    .map((item) => '<div class="stat"><strong>' + item[0] + '</strong><small>' + item[1] + '</small></div>').join("");
}
function renderKinds() {
  $("#kind-grid").innerHTML = ["markdown", "json", "sqlite", "binary"].map((kind) =>
    '<div class="kind-card"><span>' + icons[kind] + '</span><strong>' + labels[kind] + '</strong><small>' + (state.stats.kinds[kind] || 0).toLocaleString() + ' 个文件</small></div>'
  ).join("");
}
function visibleEntries() {
  if (state.filter === "all") return state.entries;
  if (state.filter === "hidden") return state.entries.filter((entry) => entry.hidden);
  return state.entries.filter((entry) => entry.readable);
}
function makeTree(entries) {
  const root = { path: "", dirs: new Map(), files: [] };
  for (const entry of entries) {
    const parts = entry.path.split("/"); let node = root;
    for (const part of parts.slice(0, -1)) {
      const path = node.path ? node.path + "/" + part : part;
      if (!node.dirs.has(part)) node.dirs.set(part, { name: part, path, dirs: new Map(), files: [] });
      node = node.dirs.get(part);
    }
    node.files.push(entry);
  }
  return root;
}
function descendantCount(node) {
  let count = node.files.length; for (const child of node.dirs.values()) count += descendantCount(child); return count;
}
function renderNodes(node, depth = 0) {
  const output = [];
  for (const dir of [...node.dirs.values()].sort((a, b) => a.name.localeCompare(b.name, "zh-CN"))) {
    const open = state.expanded.has(dir.path);
    output.push('<div class="tree-row" data-dir="' + esc(dir.path) + '" style="padding-left:' + (depth * 13 + 4) + 'px"><button class="twisty">' + (open ? "⌄" : "›") + '</button><span class="file-icon">' + (open ? "▾" : "▸") + '</span><span class="node-name">' + esc(dir.name) + '</span><span class="node-count">' + descendantCount(dir) + '</span></div>');
    if (open) output.push(renderNodes(dir, depth + 1));
  }
  for (const file of [...node.files].sort((a, b) => a.name.localeCompare(b.name, "zh-CN"))) {
    output.push('<div class="tree-row ' + (file.path === state.selected ? "active" : "") + '" data-path="' + esc(file.path) + '" title="' + esc(file.path) + '" style="padding-left:' + (depth * 13 + 27) + 'px"><span class="file-icon">' + icons[file.kind] + '</span><span class="node-name">' + esc(file.name) + '</span></div>');
  }
  return output.join("");
}
function renderTree() { $("#tree").innerHTML = renderNodes(makeTree(visibleEntries())); }
function showView(name) { ["welcome", "reader", "search-results"].forEach((id) => $("#" + id).classList.toggle("hidden", id !== name)); }

async function openFile(path, push = true) {
  state.selected = path; renderTree(); showView("reader"); $("#reader-body").innerHTML = '<div class="empty-state">正在读取…</div>';
  if (push) history.pushState(null, "", "#/" + encodeURIComponent(state.source) + "/file/" + encodeURIComponent(path));
  const entry = state.entries.find((item) => item.path === path); if (!entry) return;
  renderHeader(entry);
  try {
    if (entry.kind === "sqlite") await renderSqlite(entry);
    else { const data = await api("/api/file?" + sourceQuery({ path })); renderContent(data.entry, data.content); }
  } catch (error) {
    if (error.data && error.data.stale) {
      await loadIndex();
      state.selected = "";
      $("#reader-body").innerHTML = '<div class="empty-state">这个文件已经不在当前数据源里，目录索引已刷新。</div>';
    } else {
      $("#reader-body").innerHTML = '<div class="empty-state">读取失败：' + esc(error.message) + '</div>';
    }
  }
  $("#sidebar").classList.remove("open");
}
function renderHeader(entry) {
  const parts = entry.path.split("/");
  $("#breadcrumbs").innerHTML = [state.stats.root, ...parts.slice(0, -1)].map((part) => "<span>" + esc(part) + "</span>").join("");
  $("#file-kind").textContent = labels[entry.kind]; $("#file-title").textContent = entry.name;
  $("#file-meta").innerHTML = "<span>" + formatBytes(entry.size) + "</span><span>修改于 " + new Date(entry.modified).toLocaleString("zh-CN") + "</span><span>" + esc(entry.path) + "</span>";
}
function inlineMarkdown(text) {
  return text.replace(/\x60([^\x60]+)\x60/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>").replace(/__([^_]+)__/g, "<strong>$1</strong>").replace(/\*([^*]+)\*/g, "<em>$1</em>").replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}
function markdown(text) {
  const fence = String.fromCharCode(96, 96, 96);
  return esc(text).replace(/\r\n?/g, "\n").split(fence).map((block, index) => {
    if (index % 2) return "<pre><code>" + block.replace(/^\w+\n/, "") + "</code></pre>";
    let html = "", list = null, quote = false;
    const closeList = () => { if (list) { html += "</" + list + ">"; list = null; } };
    for (const line of block.split("\n")) {
      const heading = line.match(/^(#{1,6})\s+(.+)/), item = line.match(/^\s*([-*+] |\d+\. )(.+)/);
      if (heading) { closeList(); if (quote) { html += "</blockquote>"; quote = false; } const level = heading[1].length; html += "<h" + level + ">" + inlineMarkdown(heading[2]) + "</h" + level + ">"; }
      else if (item) { const type = /^\d/.test(item[1]) ? "ol" : "ul"; if (list !== type) { closeList(); html += "<" + type + ">"; list = type; } html += "<li>" + inlineMarkdown(item[2]) + "</li>"; }
      else if (line.startsWith("&gt; ")) { closeList(); if (!quote) { html += "<blockquote>"; quote = true; } html += "<p>" + inlineMarkdown(line.slice(5)) + "</p>"; }
      else if (!line.trim()) { closeList(); if (quote) { html += "</blockquote>"; quote = false; } }
      else { closeList(); if (quote) { html += "</blockquote>"; quote = false; } html += "<p>" + inlineMarkdown(line) + "</p>"; }
    }
    closeList(); if (quote) html += "</blockquote>"; return html;
  }).join("");
}
function renderContent(entry, content) {
  const body = $("#reader-body");
  if (content === null) body.innerHTML = '<div class="binary-card"><div class="big-icon">' + icons[entry.kind] + "</div><h2>" + labels[entry.kind] + '文件</h2><p class="lead">该文件不适合在浏览器中直接显示。阅读器仍保留路径、大小和更新时间，且不会修改原文件。</p></div>';
  else if (!content) body.innerHTML = '<div class="empty-state">这是一个空文件。</div>';
  else if (entry.kind === "markdown") body.innerHTML = '<div class="prose">' + markdown(content) + "</div>";
  else body.innerHTML = '<pre class="code-view">' + esc(content) + "</pre>";
}
async function renderSqlite(entry, table = null, page = 1) {
  const query = sourceQuery({ path: entry.path, page }); if (table) query.set("table", table);
  const data = await api("/api/sqlite?" + query);
  const chips = data.tables.map((name) => '<button class="table-chip ' + (name === data.table ? "active" : "") + '" data-table="' + esc(name) + '">' + esc(name) + "</button>").join("");
  let html = '<div class="database-toolbar"><span class="eyebrow">TABLES</span>' + (chips || '<span class="lead">没有用户表</span>') + "</div>";
  if (data.table) {
    const head = data.columns.map((column) => "<th>" + esc(column) + "</th>").join("");
    const rows = data.rows.map((row) => "<tr>" + row.map((value) => "<td>" + esc(value === null ? "NULL" : value) + "</td>").join("") + "</tr>").join("");
    const pages = Math.max(1, Math.ceil(data.total / data.limit));
    html += '<div class="table-wrap"><table class="data-table"><thead><tr>' + head + "</tr></thead><tbody>" + rows + '</tbody></table></div><div class="pager"><span>' + data.total.toLocaleString() + " 行 · " + data.page + "/" + pages + ' 页</span><button class="table-chip" data-page="' + (data.page - 1) + '" ' + (data.page <= 1 ? "disabled" : "") + '>上一页</button><button class="table-chip" data-page="' + (data.page + 1) + '" ' + (data.page >= pages ? "disabled" : "") + ">下一页</button></div>";
  } else if (data.tables.length) html += '<div class="empty-state">选择一张表查看数据。数据库以 SQLite 只读模式打开。</div>';
  $("#reader-body").innerHTML = html;
  $("#reader-body").querySelectorAll("[data-table]").forEach((node) => node.onclick = () => renderSqlite(entry, node.dataset.table, 1));
  $("#reader-body").querySelectorAll("[data-page]").forEach((node) => node.onclick = () => !node.disabled && renderSqlite(entry, data.table, Number(node.dataset.page)));
}
async function runSearch(query) {
  query = query.trim(); if (!query) { history.pushState(null, "", "#/" + encodeURIComponent(state.source)); showView("welcome"); return; }
  showView("search-results"); $("#result-title").textContent = "“" + query + "”"; $("#result-count").textContent = "搜索中…";
  const data = await api("/api/search?" + sourceQuery({ q: query })), results = data.results;
  $("#result-count").textContent = results.length + " 个结果" + (results.length === 200 ? "（已截断）" : "");
  $("#result-list").innerHTML = results.length ? results.map((entry) => '<div class="result-item" data-path="' + esc(entry.path) + '"><span class="result-icon">' + icons[entry.kind] + "</span><div><h3>" + esc(entry.name) + '</h3><div class="result-path">' + esc(entry.path) + "</div>" + (entry.snippet ? '<p class="result-snippet">' + esc(entry.snippet) + "</p>" : "") + '</div><span class="result-size">' + formatBytes(entry.size) + "</span></div>").join("") : '<div class="empty-state">没有找到匹配内容，试试更短的关键词。</div>';
  $("#result-list").querySelectorAll("[data-path]").forEach((node) => node.onclick = () => openFile(node.dataset.path));
}
async function route() {
  const match = location.hash.match(/^#\/([^/]+)\/file\/(.+)$/), sourceMatch = location.hash.match(/^#\/([^/]+)$/);
  const routeSource = decodeURIComponent(match ? match[1] : sourceMatch ? sourceMatch[1] : "");
  if (routeSource && routeSource !== state.source && state.sources.some((source) => source.id === routeSource)) {
    state.source = routeSource; $("#source-select").value = state.source; renderSourceDescription(); await loadIndex();
  }
  if (match) openFile(decodeURIComponent(match[2]), false); else showView("welcome");
}

$("#tree").addEventListener("click", (event) => { const row = event.target.closest(".tree-row"); if (!row) return; if (row.dataset.dir !== undefined) { state.expanded.has(row.dataset.dir) ? state.expanded.delete(row.dataset.dir) : state.expanded.add(row.dataset.dir); renderTree(); } if (row.dataset.path) openFile(row.dataset.path); });
document.querySelectorAll(".filter").forEach((button) => button.onclick = () => { document.querySelectorAll(".filter").forEach((node) => node.classList.remove("active")); button.classList.add("active"); state.filter = button.dataset.filter; renderTree(); });
$("#search").addEventListener("input", (event) => { clearTimeout(state.timer); state.timer = setTimeout(() => runSearch(event.target.value), 220); });
$("#source-select").onchange = async (event) => { state.source = event.target.value; localStorage.setItem("data-reader-source", state.source); state.selected = ""; state.entries = []; $("#search").value = ""; renderSourceDescription(); history.pushState(null, "", "#/" + encodeURIComponent(state.source)); showView("welcome"); await loadIndex(); toast("已切换数据源"); };
$("#refresh").onclick = async () => { $("#refresh").textContent = "…"; await api("/api/refresh?" + sourceQuery()); await loadIndex(); $("#refresh").textContent = "↻"; toast("索引已刷新"); };
$("#copy-path").onclick = () => navigator.clipboard.writeText(state.selected).then(() => toast("路径已复制"));
$("#theme-toggle").onclick = () => { const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = theme; localStorage.setItem("data-reader-theme", theme); };
$("#nav-toggle").onclick = () => $("#sidebar").classList.toggle("open");
document.addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); $(".search-box").classList.add("open"); $("#search").focus(); } if (event.key === "Escape") { $("#search").value = ""; $(".search-box").classList.remove("open"); history.pushState(null, "", "#/" + encodeURIComponent(state.source)); showView("welcome"); } });
window.addEventListener("hashchange", route);
document.documentElement.dataset.theme = localStorage.getItem("data-reader-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
loadSources().then(loadIndex).then(route).catch((error) => { $("#welcome").innerHTML = '<div class="empty-state">无法加载数据索引：' + esc(error.message) + "</div>"; });
