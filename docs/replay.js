/* Replays traces written by bench/recorder.py. */
(function () {
  "use strict";
  var BENCHES = [
    { id: "bbh", name: "BBH", setting: "cooperative" },
    { id: "humaneval", name: "HumanEval", setting: "cooperative" },
    { id: "sotopia", name: "SOTOPIA", setting: "mixed-motive" },
    { id: "deliberation", name: "LLM-Deliberation", setting: "mixed-motive" },
    { id: "werewolf", name: "Werewolf", setting: "adversarial" }
  ];
  var SVGNS = "http://www.w3.org/2000/svg";
  var $ = function (id) { return document.getElementById(id); };
  var reduceMotion = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

  var traces = {};          // id -> trace or null
  var cur = null;           // { trace, bench, events: [...flattened], states: [...], epStart: [] }
  var pos = 0, playing = false, timer = null, selected = null, layout = null;

  /* ---------------- loading ---------------- */
  fetch("runs/index.json").then(function (r) { return r.ok ? r.json() : { traces: [] }; }).catch(function () { return { traces: [] }; })
  .then(function (idx) {
    var have = {}; (idx.traces || []).forEach(function (id) { have[id] = 1; });
    return Promise.all(BENCHES.map(function (b) {
      if (!have[b.id]) { traces[b.id] = null; return null; }
      return fetch("runs/" + b.id + ".json").then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; })
        .then(function (t) { traces[b.id] = t; });
    }));
  }).then(function () {
    renderTabs();
    var want = (location.hash || "").replace("#", "");
    var first = BENCHES.filter(function (b) { return traces[b.id]; })[0];
    if (!first) { $("about").innerHTML = '<div class="empty-state">No recordings are published yet.</div>'; return; }
    choose(traces[want] ? want : first.id, true);
  });

  window.addEventListener("hashchange", function () {
    var id = (location.hash || "").replace("#", "");
    if (traces[id] && (!cur || cur.bench.id !== id)) choose(id, false);
  });

  function renderTabs() {
    $("tabs").innerHTML = "";
    BENCHES.forEach(function (b) {
      var t = document.createElement("button");
      t.type = "button"; t.className = "tab" + (traces[b.id] ? "" : " missing"); t.setAttribute("role", "tab");
      t.setAttribute("data-setting", b.setting); t.setAttribute("data-id", b.id); t.setAttribute("aria-selected", "false");
      t.innerHTML = "<span>" + b.setting + "</span><b>" + b.name + "</b>";
      if (traces[b.id]) t.onclick = function () { choose(b.id, false); };
      else t.title = "Recording not published yet";
      $("tabs").appendChild(t);
    });
  }

  /* ---------------- choose a benchmark ---------------- */
  function choose(id, autoplay) {
    stop();
    var trace = traces[id];
    var bench = BENCHES.filter(function (b) { return b.id === id; })[0];
    document.querySelectorAll(".tab").forEach(function (t) { t.setAttribute("aria-selected", String(t.getAttribute("data-id") === id)); });
    try { history.replaceState(null, "", "#" + id); } catch (e) {}
    var events = [], epStart = [];
    trace.episodes.forEach(function (ep, ei) {
      epStart.push(events.length);
      ep.events.forEach(function (ev) { events.push(Object.assign({ ep: ei }, ev)); });
    });
    // agent state after each event (carried across episodes)
    var states = [], st = {};
    events.forEach(function (ev) {
      if (ev.t === "hook" && ev.agent) { st = Object.assign({}, st); st[ev.agent] = ev.state; }
      states.push(st);
    });
    cur = { trace: trace, bench: bench, events: events, states: states, epStart: epStart };
    selected = (trace.agents[0] || {}).id;
    renderAbout(); renderEpisodes(); buildStage(); renderPicker();
    go(0, true);
    drawScrub();
    if (autoplay && !reduceMotion) play();
  }

  function renderAbout() {
    var t = cur.trace, src = t.source || {}, m = t.models || {};
    var facts = [];
    Object.keys(src).forEach(function (k) {
      if (k === "description" || k === "notes") return;
      facts.push([humanize(k), esc(flat(src[k]))]);
    });
    Object.keys(m).forEach(function (k) { facts.push([humanize(k) + " model", '<code>' + esc(m[k]) + '</code>']); });
    facts.push(["Recorded", esc((t.recorded_at || "").slice(0, 10))]);
    if (t.usage && t.usage.calls) facts.push(["Model calls", t.usage.calls.toLocaleString() + (t.usage.input_tokens ? " · " + Math.round((t.usage.input_tokens + t.usage.output_tokens) / 1000).toLocaleString() + "k tokens" : "")]);
    $("about").innerHTML =
      '<div><h2>' + esc(t.title || cur.bench.name) + '</h2>' +
      (src.description ? '<p>' + esc(src.description) + '</p>' : '') +
      '<p>Recorded with the <code>evolvemas</code> plug-in on a real ' + esc(cur.bench.name) + ' task. This is a demonstration run, not one of the runs reported in the paper.' + (src.notes ? ' ' + esc(src.notes) : '') + '</p></div>' +
      '<dl class="facts">' + facts.map(function (f) { return '<dt>' + f[0] + '</dt><dd>' + f[1] + '</dd>'; }).join("") + '</dl>';
  }

  function renderEpisodes() {
    var box = $("episodes");
    box.innerHTML = '<span class="lab">Episode</span>';
    cur.trace.episodes.forEach(function (ep, i) {
      var b = document.createElement("button");
      b.type = "button"; b.className = "ep"; b.setAttribute("aria-pressed", "false"); b.setAttribute("data-ep", i);
      b.innerHTML = (i + 1) + '<small>' + esc(resultShort(ep.result)) + '</small>';
      b.onclick = function () { stop(); go(cur.epStart[i], true); };
      box.appendChild(b);
    });
  }

  function resultShort(r) {
    if (!r) return "";
    if (r.summary) return r.summary;
    if ("agreeing_parties" in r) return r.agreeing_parties + "/" + (r.of || 6) + " agree";
    if (r.deal && typeof r.deal === "object" && "reached" in r.deal) {
      var g = r.scores && typeof r.scores === "object" ? Object.keys(r.scores).map(function (k) { var s = r.scores[k]; return k.split(" ")[0] + " " + (s && typeof s === "object" ? s.goal : s); }) : [];
      return (r.deal.reached ? "deal $" + r.deal.price : "no deal") + (g.length ? " · goal " + g.join(", ") : "");
    }
    if ("correct" in r) return r.correct ? "correct" : "wrong";
    if ("passed" in r) return r.passed ? "passed" : "failed";
    if ("winner" in r) return r.winner + " win";
    if ("agreeing" in r) return r.agreeing + "/6 agree";
    var k = Object.keys(r)[0]; return k ? k + " " + r[k] : "";
  }

  /* ---------------- stage ---------------- */
  function buildStage() {
    var svg = $("stage"); svg.innerHTML = "";
    var agents = cur.trace.agents, n = agents.length, cx = 400, cy = 300;
    var rx = n <= 2 ? 250 : 255, ry = n <= 2 ? 0 : 190;
    layout = { env: { x: cx, y: cy }, a: {} };
    var g = el("g", { "class": "links" }); svg.appendChild(g);
    agents.forEach(function (a, i) {
      var ang = n === 2 ? (i === 0 ? Math.PI : 0) : (-Math.PI / 2 + i * 2 * Math.PI / n);
      var x = cx + rx * Math.cos(ang), y = cy + (n === 2 ? 0 : ry * Math.sin(ang));
      var ux = Math.cos(ang), uy = n === 2 ? 0 : Math.sin(ang);
      var nx = -uy, ny = ux;                          // tangent
      var mx = x + ux * 74, my = y + uy * 64;         // MemoryAgent, outward
      if (n === 2) { mx = x; my = y - 96; nx = 1; ny = 0; }
      var p = { x: x, y: y, mx: mx, my: my, ux: ux, uy: uy,
                Mx: mx - 18, My: my - 34, Kx: mx + 18, Ky: my - 34 };
      if (n !== 2) { p.Mx = mx + ux * 34 - nx * 18; p.My = my + uy * 30 - ny * 18; p.Kx = mx + ux * 34 + nx * 18; p.Ky = my + uy * 30 + ny * 18; }
      layout.a[a.id] = p;
      g.appendChild(el("line", { x1: x, y1: y, x2: mx, y2: my, "class": "link" }));
      g.appendChild(el("line", { x1: mx, y1: my, x2: p.Mx, y2: p.My, "class": "link" }));
      g.appendChild(el("line", { x1: mx, y1: my, x2: p.Kx, y2: p.Ky, "class": "link" }));
      g.appendChild(el("line", { x1: x, y1: y, x2: cx, y2: cy, "class": "link", "stroke-dasharray": "2 5" }));
    });
    // environment
    var envG = el("g", {});
    envG.appendChild(el("rect", { x: cx - 62, y: cy - 30, width: 124, height: 60, rx: 10, "class": "envnode", id: "envbox" }));
    var et = el("text", { x: cx, y: cy - 6, "text-anchor": "middle", "class": "node-label" }); et.textContent = "Environment"; envG.appendChild(et);
    var es = el("text", { x: cx, y: cy + 13, "text-anchor": "middle", "class": "node-role", id: "envsub" }); es.textContent = envLabel(); envG.appendChild(es);
    svg.appendChild(envG);
    agents.forEach(function (a) {
      var p = layout.a[a.id], gg = el("g", { "data-agent": a.id, style: "cursor:pointer" });
      gg.appendChild(el("rect", { x: p.Mx - 11, y: p.My - 9, width: 22, height: 18, rx: 3, "class": "store", id: "M-" + a.id }));
      var tm = el("text", { x: p.Mx, y: p.My + 3, "text-anchor": "middle", "class": "store-label" }); tm.textContent = "M"; gg.appendChild(tm);
      gg.appendChild(el("rect", { x: p.Kx - 11, y: p.Ky - 9, width: 22, height: 18, rx: 3, "class": "store", id: "K-" + a.id }));
      var tk = el("text", { x: p.Kx, y: p.Ky + 3, "text-anchor": "middle", "class": "store-label" }); tk.textContent = "K"; gg.appendChild(tk);
      gg.appendChild(el("circle", { cx: p.mx, cy: p.my, r: 13, "class": "ma" }));
      var tma = el("text", { x: p.mx, y: p.my + 3, "text-anchor": "middle", "class": "ma-label" }); tma.textContent = "MA"; gg.appendChild(tma);
      gg.appendChild(el("circle", { cx: p.x, cy: p.y, r: 30, "class": "node-main", id: "N-" + a.id }));
      var ti = el("text", { x: p.x, y: p.y + 5, "text-anchor": "middle", "class": "node-label", style: "font-size:15px" }); ti.textContent = initials(a.label || a.id); gg.appendChild(ti);
      // name and role sit on the side facing the centre (below the node when there are two agents)
      var lx, ly, anchor = "middle";
      if (n === 2) { lx = p.x; ly = p.y + 50; }
      else {
        lx = p.x - p.ux * 46; ly = p.y - p.uy * 44 + 4;
        if (Math.abs(p.ux) > 0.35) anchor = p.ux > 0 ? "end" : "start";
        if (Math.abs(p.uy) > 0.8) ly = p.y - p.uy * 50 + (p.uy > 0 ? -12 : 12);
      }
      var tl = el("text", { x: lx, y: ly, "text-anchor": anchor, "class": "node-label" }); tl.textContent = short(a.label || a.id, 22); gg.appendChild(tl);
      var tr = el("text", { x: lx, y: ly + 14, "text-anchor": anchor, "class": "node-role" }); tr.textContent = short(a.role || "", 24); gg.appendChild(tr);
      var title = el("title", {}); title.textContent = (a.label || a.id) + (a.role ? " · " + a.role : "") + " — click to inspect its memory"; gg.appendChild(title);
      gg.addEventListener("click", function () { selected = a.id; renderPicker(); renderAgent(); markSelected(); });
      svg.appendChild(gg);
    });
    var fx = el("g", { id: "fx" }); svg.appendChild(fx);
    markSelected();
  }
  function envLabel() {
    var b = cur.bench.id;
    return { bbh: "question · answer key", humaneval: "tests", sotopia: "scenario · judge", deliberation: "game · scores", werewolf: "night · votes" }[b] || "";
  }
  function markSelected() {
    cur.trace.agents.forEach(function (a) { var n = $("N-" + a.id); if (n) n.classList.toggle("selected", a.id === selected); });
  }

  function fly(x1, y1, x2, y2, color, dashed, dur) {
    var fx = $("fx"); if (!fx) return;
    var line = el("line", { x1: x1, y1: y1, x2: x2, y2: y2, stroke: color, "stroke-width": 2.2, opacity: 0.85 });
    if (dashed) line.setAttribute("stroke-dasharray", "6 5");
    var dot = el("circle", { r: 6, fill: color, cx: x1, cy: y1 });
    fx.appendChild(line); fx.appendChild(dot);
    var t0 = performance.now();
    if (reduceMotion) { dot.setAttribute("cx", x2); dot.setAttribute("cy", y2); }
    (function step(now) {
      var k = Math.min(1, (now - t0) / dur);
      if (!reduceMotion) { dot.setAttribute("cx", x1 + (x2 - x1) * k); dot.setAttribute("cy", y1 + (y2 - y1) * k); }
      line.setAttribute("opacity", String(0.85 * (1 - Math.max(0, k - 0.6) / 0.4)));
      if (k < 1) requestAnimationFrame(step); else { line.remove(); dot.remove(); }
    })(t0);
  }
  function flash(id, cls, dur) {
    var n = $(id); if (!n) return;
    n.classList.add(cls); setTimeout(function () { n.classList.remove(cls); }, dur);
  }
  function animate(ev) {
    var A = layout.a, E = layout.env, d = Math.min(700, speed() * 0.7);
    var C = { fg: "var(--fg)", priv: "var(--private)", within: "var(--within)", cross: "var(--cross)", env: "var(--env)" };
    document.querySelectorAll(".node-main.active").forEach(function (n) { n.classList.remove("active"); });
    var fxg = $("fx"); if (fxg) fxg.innerHTML = "";
    if (ev.t === "msg") {
      var from = A[ev.from] || E, isPriv = ev.channel === "private", col = ev.channel === "system" ? C.env : isPriv ? C.priv : C.fg;
      (ev.to || []).forEach(function (to) { var q = A[to] || E; fly(from.x, from.y, q.x, q.y, col, isPriv, d); });
      if (A[ev.from]) $("N-" + ev.from).classList.add("active");
    } else if (ev.t === "act") {
      if (A[ev.agent]) $("N-" + ev.agent).classList.add("active");
    } else if (ev.t === "env") {
      var vis = (ev.data && ev.data.visible_to) || [];
      vis.forEach(function (to) { var q = A[to]; if (q) fly(E.x, E.y, q.x, q.y, C.env, true, d); });
      flash("envbox", "active", d);
    } else if (ev.t === "hook") {
      var p = A[ev.agent]; if (!p) return;
      if (ev.name === "context") { fly(p.Mx, p.My, p.x, p.y, C.within, false, d); fly(p.Kx, p.Ky, p.x, p.y, C.cross, false, d); }
      else if (ev.name === "observe") { fly(p.x, p.y, p.mx, p.my, C.within, false, d / 2); setTimeout(function () { fly(p.mx, p.my, p.Mx, p.My, C.within, false, d / 2); flash("M-" + ev.agent, "flash-m", d); }, d / 2); }
      else if (ev.name === "end_episode") { fly(E.x, E.y, p.mx, p.my, C.env, false, d / 2); setTimeout(function () { fly(p.mx, p.my, p.Kx, p.Ky, C.cross, false, d / 2); flash("K-" + ev.agent, "flash-k", d); }, d / 2); }
      else if (ev.name === "begin_episode") { flash("M-" + ev.agent, "flash-m", d / 2); }
    } else if (ev.t === "result") {
      cur.trace.agents.forEach(function (a) { var q = A[a.id]; fly(E.x, E.y, q.x, q.y, C.env, false, d); });
    }
  }

  /* ---------------- playback ---------------- */
  function speed() { return +$("speed").value; }
  function visible(ev) {
    var f = $("filter").value;
    if (f === "all") return true;
    if (f === "talk") return ev.t === "msg" || ev.t === "result" || ev.t === "env";
    return ev.t === "hook" && (ev.name === "observe" || ev.name === "end_episode");
  }
  function nextIdx(i, dir) {
    for (var j = i + dir; j >= 0 && j < cur.events.length; j += dir) if (visible(cur.events[j])) return j;
    return -1;
  }
  function go(i, quiet) {
    pos = Math.max(0, Math.min(cur.events.length - 1, i));
    var ev = cur.events[pos];
    if (!quiet) animate(ev);
    if (ev.t === "hook" && ev.agent && ev.agent !== selected && (ev.name === "observe" || ev.name === "end_episode" || ev.name === "context")) { selected = ev.agent; renderPicker(); markSelected(); }
    else if ((ev.t === "msg" || ev.t === "act") && (ev.from || ev.agent) && $("N-" + (ev.from || ev.agent))) { /* keep selection */ }
    renderNow(ev); renderAgent();
    $("pos").textContent = "Episode " + (ev.ep + 1) + " · step " + (pos - cur.epStart[ev.ep] + 1) + " of " + cur.trace.episodes[ev.ep].events.length;
    document.querySelectorAll(".ep").forEach(function (b) { b.setAttribute("aria-pressed", String(+b.getAttribute("data-ep") === ev.ep)); });
    drawScrub();
  }
  function play() {
    if (playing) return;
    playing = true; $("play").textContent = "Pause";
    (function tick() {
      var j = nextIdx(pos, 1);
      if (j < 0) { stop(); return; }
      go(j);
      timer = setTimeout(tick, speed());
    })();
  }
  function stop() { playing = false; clearTimeout(timer); var b = $("play"); if (b) b.textContent = "Play"; }
  $("play").onclick = function () { playing ? stop() : play(); };
  $("next").onclick = function () { stop(); var j = nextIdx(pos, 1); if (j >= 0) go(j); };
  $("prev").onclick = function () { stop(); var j = nextIdx(pos, -1); if (j >= 0) go(j); };
  $("filter").onchange = function () { if (!visible(cur.events[pos])) { var j = nextIdx(pos, 1); if (j >= 0) go(j, true); } drawScrub(); };
  document.addEventListener("keydown", function (e) {
    if (!cur || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
    if (e.key === "ArrowRight") { $("next").click(); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { $("prev").click(); e.preventDefault(); }
    else if (e.key === " " && document.activeElement === document.body) { $("play").click(); e.preventDefault(); }
  });

  /* scrubber */
  function colorOf(ev) {
    var cs = getComputedStyle(document.documentElement);
    if (ev.t === "msg") return cs.getPropertyValue(ev.channel === "private" ? "--private" : "--fg");
    if (ev.t === "hook") return cs.getPropertyValue(ev.name === "end_episode" ? "--cross" : "--within");
    if (ev.t === "env" || ev.t === "result") return cs.getPropertyValue("--env");
    return cs.getPropertyValue("--rule");
  }
  function drawScrub() {
    if (!cur) return;
    var cv = $("scrubcv"), r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    cv.width = Math.max(1, r.width * dpr); cv.height = Math.max(1, r.height * dpr);
    var c = cv.getContext("2d"), n = cur.events.length, w = cv.width, h = cv.height;
    c.clearRect(0, 0, w, h);
    var cs = getComputedStyle(document.documentElement);
    for (var i = 0; i < n; i++) {
      var ev = cur.events[i], x = (i + 0.5) / n * w;
      c.globalAlpha = visible(ev) ? 0.9 : 0.15;
      c.fillStyle = colorOf(ev);
      var hh = ev.t === "hook" && ev.name === "end_episode" ? h * 0.9 : ev.t === "msg" ? h * 0.7 : h * 0.45;
      c.fillRect(x - Math.max(1, w / n / 3), (h - hh) / 2, Math.max(1.5 * dpr, w / n * 0.6), hh);
    }
    c.globalAlpha = 1; c.fillStyle = cs.getPropertyValue("--muted");
    cur.epStart.forEach(function (s, k) { if (k) c.fillRect(s / n * w - dpr, 0, 2 * dpr, h); });
    c.fillStyle = cs.getPropertyValue("--fg");
    var px = (pos + 0.5) / n * w; c.fillRect(px - 1.5 * dpr, 0, 3 * dpr, h);
    $("scrub").setAttribute("aria-valuenow", pos); $("scrub").setAttribute("aria-valuemax", n - 1);
  }
  function scrubTo(e) {
    var r = $("scrub").getBoundingClientRect(), k = (e.clientX - r.left) / r.width;
    stop(); go(Math.round(k * (cur.events.length - 1)), true);
  }
  var dragging = false;
  $("scrub").addEventListener("pointerdown", function (e) { dragging = true; scrubTo(e); });
  window.addEventListener("pointermove", function (e) { if (dragging) scrubTo(e); });
  window.addEventListener("pointerup", function () { dragging = false; });
  $("scrub").addEventListener("keydown", function (e) { if (e.key === "ArrowRight") { stop(); go(pos + 1); } if (e.key === "ArrowLeft") { stop(); go(pos - 1); } });
  window.addEventListener("resize", drawScrub);

  /* ---------------- inspector ---------------- */
  function name(id) { var a = cur.trace.agents.filter(function (x) { return x.id === id; })[0]; return a ? (a.label || a.id) : id; }
  function names(ids) {
    var all = cur.trace.agents.map(function (a) { return a.id; });
    var others = ids.filter(function (x) { return x !== "env"; });
    if (others.length > 1 && others.length >= all.length - 1) return "everyone";
    return others.map(name).join(", ") || "environment";
  }
  function prevState(agent) {
    for (var j = pos - 1; j >= 0; j--) { var s = cur.states[j][agent]; if (s) return s; }
    return null;
  }
  function changes(before, after) {
    var out = [];
    if (!after) return out;
    var b = before || { episodic: [], semantic: {}, working: "", strategies: [], partners: {} };
    var oldE = {}; b.episodic.forEach(function (e) { oldE[e.summary] = 1; });
    var ne = after.episodic.filter(function (e) { return !oldE[e.summary]; }).length;
    var ns = 0, cs = 0; Object.keys(after.semantic).forEach(function (k) { if (!b.semantic[k]) ns++; else if (b.semantic[k].value !== after.semantic[k].value) cs++; });
    var dropped = Object.keys(b.semantic).filter(function (k) { return !after.semantic[k]; }).length;
    if (ne) out.push(ne + " episodic " + (ne > 1 ? "entries" : "entry") + " added");
    if (ns) out.push(ns + " fact" + (ns > 1 ? "s" : "") + " added");
    if (cs) out.push(cs + " fact" + (cs > 1 ? "s" : "") + " revised");
    if (dropped) out.push(dropped + " episode-only fact" + (dropped > 1 ? "s" : "") + " cleared");
    if ((after.working || "") !== (b.working || "")) out.push(after.working ? "working memory rewritten" : "working memory cleared");
    var oldS = {}; b.strategies.forEach(function (s) { oldS[s] = 1; });
    var nst = after.strategies.filter(function (s) { return !oldS[s]; }).length;
    var kept = after.strategies.length - nst, droppedS = b.strategies.length - kept;
    if (nst) out.push(nst + " strateg" + (nst > 1 ? "ies" : "y") + " added or rewritten");
    if (droppedS > 0) out.push(droppedS + " strateg" + (droppedS > 1 ? "ies" : "y") + " dropped");
    var np = 0; Object.keys(after.partners).forEach(function (k) { var o = b.partners[k], a = after.partners[k]; if (!o || o.tendency !== a.tendency || o.reliability !== a.reliability) np++; });
    if (np) out.push(np + " partner profile" + (np > 1 ? "s" : "") + " updated");
    return out;
  }

  function renderNow(ev) {
    var h = "";
    if (ev.t === "msg") {
      var priv = ev.channel === "private", sys = ev.channel === "system";
      h += kind(sys ? "env" : priv ? "private" : "msg", sys ? "Environment message" : priv ? "Private message" : "Message");
      h += '<div class="headline">' + esc(name(ev.from)) + ' → ' + esc(names(ev.to || [])) + '</div>';
      h += '<div class="bubble">' + esc(ev.content) + '</div>';
    } else if (ev.t === "act") {
      h += kind("msg", "Model call · " + esc(name(ev.agent)));
      h += '<div class="headline">' + esc(name(ev.agent)) + ' decides what to do. Its prompt includes the block returned by <code>context()</code>.</div>';
      h += '<div class="bubble">' + esc(ev.reply) + '</div>';
      h += raw("Full prompt sent to " + name(ev.agent), ev.prompt);
    } else if (ev.t === "env") {
      h += kind("env", "Environment");
      var vis = ev.data && ev.data.visible_to;
      h += '<div class="headline">' + esc(firstLine(ev.content)) + (vis ? ' <span class="pill" style="color:var(--private)">seen by ' + esc(names(vis)) + '</span>' : '') + '</div>';
      if (ev.content.indexOf("\n") > -1 || ev.content.length > 120) h += '<div class="bubble">' + esc(ev.content) + '</div>';
      var dd = Object.assign({}, ev.data || {}); delete dd.visible_to;
      if (Object.keys(dd).length) h += raw("Data", JSON.stringify(dd, null, 1));
    } else if (ev.t === "hook") {
      var within = ev.name !== "end_episode";
      h += kind(within ? "hook-within" : "hook-cross", ev.name + "(" + esc(name(ev.agent)) + ")");
      var ch = changes(prevState(ev.agent), ev.state);
      if (ev.name === "context") {
        h += '<div class="headline">Memory and knowledge handed to ' + esc(name(ev.agent)) + ' before it acts</div>';
        h += '<div class="bubble">' + (ev.output ? esc(ev.output) : '<span style="color:var(--muted)">Nothing yet: no memory or knowledge for this agent.</span>') + '</div>';
      } else if (ev.name === "observe") {
        var nin = (ev.input || []).length;
        h += '<div class="headline">' + esc(name(ev.agent)) + "'s MemoryAgent read " + nin + ' new item' + (nin === 1 ? '' : 's') + (ch.length ? ': ' + esc(ch.join(", ")) : ' and changed nothing') + '</div>';
        h += '<div class="bubble">' + (ev.input || []).map(function (e) { return '<b>' + esc(e.speaker) + ':</b> ' + esc(e.content); }).join("\n\n") + '</div>';
      } else if (ev.name === "end_episode") {
        h += '<div class="headline">Episode result distilled into ' + esc(name(ev.agent)) + "'s knowledge" + (ch.length ? ': ' + esc(ch.join(", ")) : '') + '</div>';
        h += raw("Result handed to end_episode()", JSON.stringify(ev.input, null, 1));
      } else if (ev.name === "begin_episode") {
        h += '<div class="headline">New episode for ' + esc(name(ev.agent)) + ': working memory and episode-only facts cleared, knowledge kept</div>';
      }
      (ev.calls || []).forEach(function (c) {
        var who = c.role === "evolve" ? "Knowledge distillation" : "MemoryAgent";
        h += raw(who + " input", c.user);
        h += raw(who + " output (JSON)", c.response);
      });
    } else if (ev.t === "result") {
      h += kind("result", "Episode result");
      h += '<div class="headline">' + esc(resultShort(ev.data)) + '</div>';
      h += raw("Result data", JSON.stringify(ev.data, null, 1), true);
    }
    $("now").innerHTML = h;
  }
  function kind(cls, label) { return '<div class="kind ' + cls + '">' + label + '</div>'; }
  function raw(label, text, open) { return '<details class="raw"' + (open ? ' open' : '') + '><summary>' + esc(label) + '</summary><pre>' + esc(text || "") + '</pre></details>'; }
  function firstLine(s) { s = String(s || ""); var i = s.indexOf("\n"); return i > -1 ? s.slice(0, i) : s; }

  function renderPicker() {
    var p = $("picker"); p.innerHTML = "";
    cur.trace.agents.forEach(function (a) {
      var b = document.createElement("button"); b.type = "button"; b.textContent = a.label || a.id;
      b.setAttribute("aria-pressed", String(a.id === selected));
      b.onclick = function () { selected = a.id; renderPicker(); renderAgent(); markSelected(); };
      p.appendChild(b);
    });
  }
  function renderAgent() {
    var s = cur.states[pos][selected], b = prevState(selected);
    var ev = cur.events[pos], justChanged = ev.t === "hook" && ev.agent === selected;
    if (!justChanged) b = s;   // highlight only on the step that changed this agent
    $("agent-title").textContent = name(selected) + " · state at this step";
    if (!s) { $("mem").innerHTML = '<div class="none">No memory yet.</div>'; return; }
    var old = b || { episodic: [], semantic: {}, working: "", strategies: [], partners: {} };
    var oe = {}; old.episodic.forEach(function (e) { oe[e.summary] = 1; });
    var os = {}; old.strategies.forEach(function (x) { os[x] = 1; });
    var h = '';
    h += '<div><h4 class="m">Episodic</h4>' + list(s.episodic.slice(-8).map(function (e) {
      return [esc(e.summary) + ' <span class="k">ep' + (e.episode + 1) + ' t' + e.turn + '</span>', !oe[e.summary] ? "new" : ""];
    }), "none") + '</div>';
    h += '<div><h4 class="m">Semantic</h4>' + list(Object.keys(s.semantic).map(function (k) {
      var v = s.semantic[k], o = old.semantic[k];
      return ['<span class="k">' + esc(k) + '</span> = ' + esc(v.value) + ' <span class="k">(' + esc(v.confidence) + (v.scope === "episode" ? ", this episode" : "") + ')</span>', !o ? "new" : o.value !== v.value ? "chg" : ""];
    }), "none") + '</div>';
    h += '<div><h4 class="m">Working</h4>' + list(s.working ? [[esc(s.working), s.working !== old.working ? "chg" : ""]] : [], "empty") + '</div>';
    h += '<div><h4 class="k">Interaction strategies</h4>' + list(s.strategies.map(function (x) { return [esc(x), !os[x] ? "new" : ""]; }), "none yet") + '</div>';
    h += '<div><h4 class="k">Partner profiles</h4>' + list(Object.keys(s.partners).map(function (k) {
      var p = s.partners[k], o = old.partners[k];
      return ['<b>' + esc(name(k)) + '</b>: ' + esc(p.tendency) + ' <span class="k">(' + esc(p.reliability) + ')</span>', !o ? "new" : (o.tendency !== p.tendency || o.reliability !== p.reliability) ? "chg" : ""];
    }), "none yet") + '</div>';
    $("mem").innerHTML = h;
  }
  function list(items, emptyText) {
    if (!items.length) return '<div class="none">' + emptyText + '</div>';
    return '<ul>' + items.map(function (it) { return '<li class="' + it[1] + '">' + it[0] + '</li>'; }).join("") + '</ul>';
  }

  /* ---------------- utils ---------------- */
  function el(tag, attrs) { var e = document.createElementNS(SVGNS, tag); for (var k in attrs) e.setAttribute(k, attrs[k]); return e; }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function initials(s) {
    s = String(s).replace(/^agent[\s_]+/i, "");
    if (s.length <= 2) return s.toUpperCase();
    var m = s.match(/^([A-Za-z])[A-Za-z_ ]*?(\d+)$/); if (m) return m[1].toUpperCase() + m[2];
    var w = s.replace(/[_-]/g, " ").split(/\s+/).filter(function (x) { return x && !/^(of|the|and)$/i.test(x); });
    if (w.length >= 2) return (w[0][0] + w[1][0]).toUpperCase();
    return s.slice(0, 2).replace(/^./, function (c) { return c.toUpperCase(); });
  }
  function humanize(k) { k = String(k).replace(/_/g, " "); return k.charAt(0).toUpperCase() + k.slice(1); }
  function flat(v) {
    if (v == null) return "";
    if (Array.isArray(v)) return v.map(flat).join("; ");
    if (typeof v === "object") return Object.keys(v).map(function (k) { return humanize(k) + ": " + flat(v[k]); }).join("; ");
    return String(v);
  }
  function short(s, n) { s = String(s); return s.length > n ? s.slice(0, n - 1) + "…" : s; }
})();
