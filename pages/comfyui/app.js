/* ComfyUI 画图视图页 —— AstrBotPluginPage bridge */
(function () {
  "use strict";

  var PAGE_SIZE = 24;
  var bridge = window.AstrBotPluginPage || window.AstrBotPluginView;

  var state = {
    history: [],
    shown: 0,
    total: 0,
    current: null,   // 当前详情条目
    status: null,
  };

  /* ---------- 基础工具 ---------- */
  function $(id) { return document.getElementById(id); }

  function withTimeout(promise, ms, message) {
    var timer;
    var timeout = new Promise(function (_, reject) {
      timer = window.setTimeout(function () { reject(new Error(message)); }, ms);
    });
    return Promise.race([promise, timeout]).finally(function () { window.clearTimeout(timer); });
  }

  function normalize(result, fallbackMessage) {
    if (result && typeof result === "object") {
      if (result.status === "error") throw new Error(result.message || fallbackMessage);
      if (Object.prototype.hasOwnProperty.call(result, "ok")) {
        if (!result.ok) throw new Error((result.error && result.error.message) || result.message || fallbackMessage);
        return result.data || {};
      }
    }
    return result || {};
  }

  function apiGet(endpoint, params) {
    return withTimeout(bridge.apiGet(endpoint, params || {}), 30000, "请求 AstrBot WebUI 超时，请刷新后重试。").then(function (r) {
      return normalize(r, "请求失败");
    });
  }

  function apiPost(endpoint, body) {
    return withTimeout(bridge.apiPost(endpoint, body || {}), 30000, "请求 AstrBot WebUI 超时，请刷新后重试。").then(function (r) {
      return normalize(r, "请求失败");
    });
  }

  var toastTimer = null;
  function toast(msg) {
    var el = $("toast");
    el.textContent = msg;
    el.hidden = false;
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(function () { el.hidden = true; }, 2400);
  }

  function fmtTime(ts) {
    var n = Number(ts || 0);
    if (!n) return "未知时间";
    var d = new Date(n * 1000);
    if (isNaN(d.getTime())) return "未知时间";
    function p(x) { return x < 10 ? "0" + x : "" + x; }
    return d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate()) + " " + p(d.getHours()) + ":" + p(d.getMinutes());
  }

  /* ---------- 状态卡 ---------- */
  function renderStatus(st) {
    state.status = st || {};
    var badge = $("statusBadge");
    var alive = st && st.comfy_alive;
    badge.textContent = alive ? "ComfyUI 已连接" : (st && st.server ? "ComfyUI 未连接" : "未检测");
    badge.className = "badge " + (alive ? "badge-ok" : "badge-bad");
    $("stServer").textContent = (st && st.server) || "—";
    $("stWorkflow").textContent = (st && st.workflow) || "—";
    $("stSize").textContent = st && st.width && st.height ? st.width + " × " + st.height : "—";
    $("stCount").textContent = st ? String(st.total_images || 0) : "—";
  }

  /* ---------- 工作流 ---------- */
  function renderWorkflows(data) {
    var box = $("workflowList");
    var items = (data && data.items) || [];
    var current = (data && data.current) || "";
    if (!items.length) {
      box.innerHTML = '<div class="empty">工作流目录为空</div>';
      return;
    }
    box.innerHTML = "";
    items.forEach(function (wf) {
      var el = document.createElement("button");
      el.className = "wf-item" + (wf.file === current ? " active" : "");
      var name = document.createElement("div");
      name.className = "wf-name";
      name.textContent = wf.file;
      var meta = document.createElement("div");
      meta.className = "wf-meta";
      meta.textContent = fmtTime(wf.mtime) + " · " + fmtSize(wf.size);
      var cur = document.createElement("div");
      cur.className = "wf-current";
      cur.textContent = wf.file === current ? "● 当前使用" : "";
      el.appendChild(name);
      el.appendChild(meta);
      if (wf.file === current) el.appendChild(cur);
      el.addEventListener("click", function () { switchWorkflow(wf.file); });
      box.appendChild(el);
    });
  }

  function fmtSize(bytes) {
    var n = Number(bytes || 0);
    if (n > 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + " MB";
    if (n > 1024) return (n / 1024).toFixed(0) + " KB";
    return n + " B";
  }

  function switchWorkflow(file) {
    if (!window.confirm("切换当前工作流为：" + file + " ？")) return;
    apiPost("page/switch-workflow", { file: file }).then(function (r) {
      toast(r.message || "已切换");
      loadStatus();
      loadWorkflows();
    }).catch(function (e) { toast("切换失败：" + (e.message || e)); });
  }

  /* ---------- 历史 ---------- */
  function loadHistory() {
    $("historyGrid").innerHTML = '<div class="empty">加载中…</div>';
    apiGet("page/history", { limit: 600 }).then(function (data) {
      state.history = (data && data.items) || [];
      state.total = (data && data.total) || state.history.length;
      state.shown = 0;
      $("historyGrid").innerHTML = "";
      $("historyMeta").textContent = "共 " + state.total + " 张";
      if (!state.history.length) {
        $("historyGrid").innerHTML = '<div class="empty">暂无出图记录</div>';
        return;
      }
      renderMore();
    }).catch(function (e) {
      $("historyGrid").innerHTML = '<div class="empty">加载失败：' + escapeHtml(e.message || e) + "</div>";
    });
  }

  function renderMore() {
    var grid = $("historyGrid");
    var slice = state.history.slice(state.shown, state.shown + PAGE_SIZE);
    slice.forEach(function (item) { grid.appendChild(historyCell(item)); });
    state.shown += slice.length;
    var btn = $("moreBtn");
    btn.hidden = state.shown >= state.history.length;
  }

  function historyCell(item) {
    var cell = document.createElement("div");
    cell.className = "h-cell";
    var img = document.createElement("img");
    img.loading = "lazy";
    img.alt = item.file || "";
    cell.appendChild(img);
    if (!item.prompt) {
      var tag = document.createElement("span");
      tag.className = "h-tag";
      tag.textContent = "无记录";
      cell.appendChild(tag);
    }
    cell.addEventListener("click", function () { openDrawer(item); });
    // 缩略图
    apiGet("page/thumb", { file: item.file }).then(function (r) {
      if (r && r.data_url) img.src = r.data_url;
      else cell.style.opacity = "0.4";
    }).catch(function () { cell.style.opacity = "0.4"; });
    return cell;
  }

  /* ---------- 详情抽屉 ---------- */
  function openDrawer(item) {
    state.current = item;
    $("dTime").textContent = fmtTime(item.ts);
    $("dPrompt").textContent = item.prompt || "（未记录，历史遗留图片）";
    $("dWorkflow").textContent = item.workflow || "—";
    $("dLora").textContent = item.loras && item.loras.length ? item.loras.join("、") : "—";
    $("dSource").textContent = item.source || "—";
    $("dUser").textContent = item.user || "—";
    var img = $("dImg");
    img.src = "";
    img.alt = item.file || "";
    apiGet("page/image", { file: item.file }).then(function (r) {
      if (r && r.data_url) {
        img.src = r.data_url;
        $("dDownload").href = r.data_url;
        $("dDownload").download = item.file || "comfyui.png";
      }
    }).catch(function (e) { toast("读取原图失败：" + (e.message || e)); });
    $("drawerMask").hidden = false;
  }

  function closeDrawer() {
    $("drawerMask").hidden = true;
    $("dImg").src = "";
    state.current = null;
  }

  function copyPrompt() {
    var text = state.current && state.current.prompt;
    if (!text) { toast("该图片没有记录提示词"); return; }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { toast("已复制"); });
    } else {
      var ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      toast("已复制");
    }
  }

  function deleteCurrent() {
    var item = state.current;
    if (!item) return;
    if (!window.confirm("删除图片 " + (item.file || "") + " 及其记录？不可恢复。")) return;
    apiPost("page/delete", { file: item.file }).then(function () {
      toast("已删除");
      closeDrawer();
      loadHistory();
      loadStatus();
    }).catch(function (e) { toast("删除失败：" + (e.message || e)); });
  }

  function escapeHtml(s) {
    return String(s || "").replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  /* ---------- 加载 ---------- */
  function loadStatus() {
    apiGet("page/status").then(renderStatus).catch(function () { renderStatus(null); });
  }
  function loadWorkflows() {
    apiGet("page/workflows").then(renderWorkflows).catch(function (e) {
      $("workflowList").innerHTML = '<div class="empty">加载失败</div>';
    });
  }
  function loadAll() {
    loadStatus();
    loadWorkflows();
    loadHistory();
  }

  /* ---------- 启动 ---------- */
  function bindEvents() {
    $("refreshBtn").addEventListener("click", loadAll);
    $("moreBtn").addEventListener("click", renderMore);
    $("closeDrawer").addEventListener("click", closeDrawer);
    $("drawerMask").addEventListener("click", function (e) {
      if (e.target === $("drawerMask")) closeDrawer();
    });
    $("copyPrompt").addEventListener("click", copyPrompt);
    $("deleteImg").addEventListener("click", deleteCurrent);
  }

  function applyTheme(ctx) {
    var dark = !!(ctx && (ctx.isDark || (ctx.theme === "dark")));
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  }

  function start() {
    if (!bridge) {
      document.body.innerHTML = '<div style="padding:24px">没有检测到 AstrBot Pages bridge，请从 AstrBot WebUI 插件页面进入。</div>';
      return;
    }
    bindEvents();
    var onCtx = bridge.onContext || bridge.onContextChange;
    if (typeof onCtx === "function") {
      try { onCtx(function (ctx) { applyTheme(ctx); }); } catch (e) { /* ignore */ }
    }
    bridge.ready().then(function (ctx) {
      if (ctx) { applyTheme(ctx); }
      loadAll();
    }).catch(function (e) {
      toast("初始化超时：" + (e.message || e));
      loadAll();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
