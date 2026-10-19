/* Compact data-flow player for the homepage hero. Plays one recorded episode from a
   bench/recorder.py trace on a loop: messages between agents, each agent's MemoryAgent
   writing memory (blue), and the result being distilled into knowledge (green). */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

  function el(tag, a) { var e = document.createElementNS(NS, tag); for (var k in a) e.setAttribute(k, a[k]); return e; }
  function txt(tag, a, s) { var e = el(tag, a); e.textContent = s; return e; }
  function clip(s, n) { s = String(s || "").replace(/\s+/g, " ").trim(); return s.length > n ? s.slice(0, n - 1) + "…" : s; }

  function mount(svg, caption, trace, epIndex, opts) {
    opts = opts || {};
    var ep = trace.episodes[epIndex], agents = trace.agents;
    var W = 520, H = 400, cx = 260, cy = 218;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.innerHTML = "";
    var P = {};
    var g0 = el("g", {}); svg.appendChild(g0);
    agents.forEach(function (a, i) {
      var ang = -Math.PI / 2 + i * 2 * Math.PI / agents.length;
      var ux = Math.cos(ang), uy = Math.sin(ang);
      var x = cx + 165 * ux, y = cy + 118 * uy;
      var mx = x + 62 * ux, my = y + 52 * uy;
      var nx = -uy, ny = ux;
      P[a.id] = { x: x, y: y, mx: mx, my: my, Mx: mx + 30 * ux - 17 * nx, My: my + 24 * uy - 17 * ny, Kx: mx + 30 * ux + 17 * nx, Ky: my + 24 * uy + 17 * ny, ux: ux, uy: uy };
    });
    // links
    agents.forEach(function (a) {
      var p = P[a.id];
      g0.appendChild(el("line", { x1: p.x, y1: p.y, x2: p.mx, y2: p.my, "class": "fl-link" }));
      g0.appendChild(el("line", { x1: p.mx, y1: p.my, x2: p.Mx, y2: p.My, "class": "fl-link" }));
      g0.appendChild(el("line", { x1: p.mx, y1: p.my, x2: p.Kx, y2: p.Ky, "class": "fl-link" }));
    });
    var env = el("g", {});
    env.appendChild(el("rect", { x: cx - 52, y: cy - 22, width: 104, height: 44, rx: 8, "class": "fl-env", id: "fl-env" }));
    env.appendChild(txt("text", { x: cx, y: cy - 3, "text-anchor": "middle", "class": "fl-label" }, "Task"));
    env.appendChild(txt("text", { x: cx, y: cy + 12, "text-anchor": "middle", "class": "fl-small" }, opts.envLabel || "answer key"));
    svg.appendChild(env);
    agents.forEach(function (a) {
      var p = P[a.id], g = el("g", {});
      g.appendChild(el("rect", { x: p.Mx - 10, y: p.My - 8, width: 20, height: 16, rx: 3, "class": "fl-store", id: "fl-M-" + a.id }));
      g.appendChild(txt("text", { x: p.Mx, y: p.My + 3.5, "text-anchor": "middle", "class": "fl-small" }, "M"));
      g.appendChild(el("rect", { x: p.Kx - 10, y: p.Ky - 8, width: 20, height: 16, rx: 3, "class": "fl-store", id: "fl-K-" + a.id }));
      g.appendChild(txt("text", { x: p.Kx, y: p.Ky + 3.5, "text-anchor": "middle", "class": "fl-small" }, "K"));
      g.appendChild(el("circle", { cx: p.mx, cy: p.my, r: 11, "class": "fl-ma" }));
      g.appendChild(txt("text", { x: p.mx, y: p.my + 3, "text-anchor": "middle", "class": "fl-tiny" }, "MA"));
      g.appendChild(el("circle", { cx: p.x, cy: p.y, r: 27, "class": "fl-agent", id: "fl-N-" + a.id }));
      g.appendChild(txt("text", { x: p.x, y: p.y + 5, "text-anchor": "middle", "class": "fl-label", style: "font-size:15px" }, (a.label || a.id).replace(/^Agent /, "")));
      g.appendChild(txt("text", { x: p.x, y: p.y + (p.uy < 0 ? 42 : -36), "text-anchor": "middle", "class": "fl-small" }, clip(a.role || "", 26)));
      svg.appendChild(g);
    });
    var fx = el("g", {}); svg.appendChild(fx);

    function fly(x1, y1, x2, y2, color, d) {
      var line = el("line", { x1: x1, y1: y1, x2: x2, y2: y2, stroke: color, "stroke-width": 2, opacity: 0.8 });
      var dot = el("circle", { r: 5, fill: color, cx: x1, cy: y1 });
      fx.appendChild(line); fx.appendChild(dot);
      var t0 = performance.now();
      (function step(now) {
        var k = reduce ? 1 : Math.min(1, (now - t0) / d);
        dot.setAttribute("cx", x1 + (x2 - x1) * k); dot.setAttribute("cy", y1 + (y2 - y1) * k);
        line.setAttribute("opacity", String(0.8 * (1 - Math.max(0, k - 0.6) / 0.4)));
        if (k < 1) requestAnimationFrame(step); else { line.remove(); dot.remove(); }
      })(t0);
    }
    function flash(id, cls, d) { var n = document.getElementById(id); if (!n) return; n.classList.add(cls); setTimeout(function () { n.classList.remove(cls); }, d); }
    function name(id) { var a = agents.filter(function (x) { return x.id === id; })[0]; return a ? a.label : id; }

    // Build the script of steps: group the per-agent hooks of a round into one step each.
    var steps = [], evs = ep.events;
    steps.push({ kind: "env", text: "A GSM8K problem goes to all three agents.", run: function (d) { agents.forEach(function (a) { fly(cx, cy, P[a.id].x, P[a.id].y, "var(--env)", d); }); } });
    for (var i = 0; i < evs.length; i++) {
      var e = evs[i];
      if (e.t === "hook" && e.name === "context") {
        var block = []; while (i < evs.length && evs[i].t === "hook" && evs[i].name === "context") block.push(evs[i++]); i--;
        var withMem = block.filter(function (b) { return b.output; }).length;
        steps.push({ kind: "mem", text: "context(): each agent's memory and knowledge go into its prompt" + (withMem ? "" : " (empty so far)") + ".",
          run: (function (blk) { return function (d) { blk.forEach(function (b) { var p = P[b.agent]; fly(p.Mx, p.My, p.x, p.y, "var(--mem)", d); fly(p.Kx, p.Ky, p.x, p.y, "var(--know)", d); }); }; })(block) });
      } else if (e.t === "msg") {
        steps.push({ kind: "msg", text: name(e.from) + " → others: " + clip(e.content, 92),
          run: (function (m) { return function (d) { var p = P[m.from]; (m.to || []).forEach(function (t) { if (P[t]) fly(p.x, p.y, P[t].x, P[t].y, "var(--fg)", d); }); var n = document.getElementById("fl-N-" + m.from); n && n.classList.add("on"); setTimeout(function () { n && n.classList.remove("on"); }, d); }; })(e) });
      } else if (e.t === "hook" && e.name === "observe") {
        var ob = []; while (i < evs.length && evs[i].t === "hook" && evs[i].name === "observe") ob.push(evs[i++]); i--;
        var call = (ob[ob.length - 1].calls || [])[0], wrote = "";
        try { var j = JSON.parse(call.response.slice(call.response.indexOf("{"), call.response.lastIndexOf("}") + 1)); wrote = (j.episodic || [])[0] || j.working || ""; } catch (err) {}
        steps.push({ kind: "mem", text: "observe(): each MemoryAgent reads the round and updates MEMORY.md" + (wrote ? " — " + name(ob[ob.length - 1].agent) + "'s wrote “" + clip(wrote, 70) + "”" : "."),
          run: (function (blk) { return function (d) { blk.forEach(function (b) { var p = P[b.agent]; fly(p.x, p.y, p.mx, p.my, "var(--mem)", d / 2); setTimeout(function () { fly(p.mx, p.my, p.Mx, p.My, "var(--mem)", d / 2); flash("fl-M-" + b.agent, "hit-m", d); }, d / 2); }); }; })(ob) });
      } else if (e.t === "result") {
        var r = e.data, c = r.answers && r.answers.agent_c;
        steps.push({ kind: "env", text: "Answer checked: team " + r.team_answer + ", key " + r.gold + (r.team_correct ? " ✓" : " ✗") + (c && !c.round1_correct ? ". Agent C's first answer (" + c.round1 + ") was wrong." : "."),
          run: function (d) { flash("fl-env", "hit-e", d); } });
      } else if (e.t === "hook" && e.name === "end_episode") {
        var en = []; while (i < evs.length && evs[i].t === "hook" && evs[i].name === "end_episode") en.push(evs[i++]); i--;
        var a0 = en.filter(function (b) { return b.agent === (opts.focus || en[0].agent); })[0] || en[0];
        var prof = a0.state.partners && a0.state.partners.agent_c;
        steps.push({ kind: "know", text: "end_episode(): the result is distilled into KNOWLEDGE.md" + (prof ? " — " + name(a0.agent) + " now rates Agent C “" + clip(prof.reliability, 60) + "”" : "."),
          run: (function (blk) { return function (d) { blk.forEach(function (b) { var p = P[b.agent]; fly(cx, cy, p.mx, p.my, "var(--env)", d / 2); setTimeout(function () { fly(p.mx, p.my, p.Kx, p.Ky, "var(--know)", d / 2); flash("fl-K-" + b.agent, "hit-k", d); }, d / 2); }); }; })(en) });
      }
    }

    var k = 0, timer = null, playing = !reduce;
    function show(s) {
      fx.innerHTML = "";
      caption.innerHTML = '<span class="fl-kind k-' + s.kind + '">' + ({ env: "task", msg: "message", mem: "memory", know: "knowledge" })[s.kind] + '</span> ' + escapeHtml(s.text);
      s.run(opts.dur || 1100);
    }
    function tick() { show(steps[k]); k = (k + 1) % steps.length; timer = setTimeout(tick, (steps[k === 0 ? steps.length - 1 : k - 1].kind === "msg" ? 2300 : 2000)); }
    function escapeHtml(s) { return String(s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
    if (playing) tick(); else show(steps[steps.length - 1]);
    return {
      toggle: function () { if (timer) { clearTimeout(timer); timer = null; return false; } tick(); return true; }
    };
  }
  window.EvolveFlow = { mount: mount };
})();
