// Admin chat client for the proprietary POST /v1/agent/chat endpoint.
//
// This replaces the old htmx-sse pair: the admin UI is now a plain client of
// the same proprietary API the eval harness uses. It POSTs a single message,
// reads the JSON-SSE stream (event: token / tool / done / error), and renders
// tokens + tool cards client-side. External file (CSP-safe); a single
// delegated submit listener survives htmx-boost navigation.
(function () {
  "use strict";

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function safeHref(url) {
    return /^https?:\/\//i.test(url || "") ? url : null;
  }

  function scrollToBottom() {
    const m = document.getElementById("chat-messages");
    if (m) m.scrollTop = m.scrollHeight;
  }

  // --- SSE framing over fetch ReadableStream ---

  async function* sseEvents(resp) {
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, i);
        buf = buf.slice(i + 2);
        const ev = parseFrame(frame);
        if (ev) yield ev;
      }
    }
    const tail = parseFrame(buf);
    if (tail) yield tail;
  }

  function parseFrame(frame) {
    let event = "message";
    const data = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    const raw = data.join("\n");
    if (!raw) return null;
    let parsed = null;
    try { parsed = JSON.parse(raw); } catch (_) { /* ignore malformed */ }
    return { event, data: parsed };
  }

  // --- DOM builders (mirror the former templ markup) ---

  function userBubble(text) {
    const wrap = el("div", "flex justify-end");
    wrap.appendChild(el("div", "max-w-2xl rounded-lg px-4 py-2 bg-blue-600 text-white text-sm", text));
    return wrap;
  }

  function assistantHost() {
    const wrap = el("div", "flex justify-start");
    const host = el("div", "max-w-2xl w-full space-y-1");
    const thinking = el("div", "flex items-center gap-1 rounded-lg px-4 py-2.5 bg-gray-100 w-fit");
    thinking.setAttribute("aria-label", "Assistant is thinking");
    thinking.innerHTML =
      '<span class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"></span>' +
      '<span class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]"></span>' +
      '<span class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]"></span>';
    const tools = el("div", "assistant-tools space-y-1");
    const bubble = el("div", "rounded-lg px-4 py-2 bg-gray-100 text-gray-900 text-sm whitespace-pre-wrap");
    bubble.hidden = true;
    host.append(thinking, tools, bubble);
    wrap.appendChild(host);
    return { wrap, thinking, tools, bubble };
  }

  const SPINNER =
    '<svg class="animate-spin h-3 w-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">' +
    '<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>' +
    '<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>';
  const CHECK =
    '<svg class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">' +
    '<path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"></path></svg>';
  const CROSS =
    '<svg class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">' +
    '<path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"></path></svg>';

  function pill(cls, iconHTML, name, durText) {
    const p = el("div", "inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-xs " + cls);
    const icon = el("span", "shrink-0");
    icon.innerHTML = iconHTML;
    p.append(icon, el("span", "font-medium", name));
    if (durText) p.appendChild(el("span", "opacity-75", durText));
    return p;
  }

  function fmtDuration(ms) {
    if (ms == null) return "";
    return ms < 1000 ? ms + "ms" : (ms / 1000).toFixed(1) + "s";
  }

  function toolCardStart(callId, name) {
    const card = el("div");
    card.dataset.callId = callId;
    card.appendChild(pill("bg-blue-50 text-blue-700", SPINNER, name || "tool", null));
    return card;
  }

  function toolCardDone(card, ev) {
    card.replaceChildren();
    const ok = !ev.error;
    card.appendChild(pill(
      ok ? "bg-green-50 text-green-700" : "bg-red-50 text-red-700",
      ok ? CHECK : CROSS,
      ev.name || "tool",
      fmtDuration(ev.duration_ms),
    ));
    if (Array.isArray(ev.citations) && ev.citations.length) {
      const ul = el("ul", "mt-1 ml-2 space-y-0.5 text-xs text-gray-700");
      for (const c of ev.citations) ul.appendChild(citationItem(c));
      card.appendChild(ul);
    }
    if (Array.isArray(ev.images) && ev.images.length) {
      const box = el("div", "mt-2 ml-2 flex flex-wrap gap-2");
      for (const img of ev.images) {
        const href = safeHref(img.url) || img.url;
        const a = el("a", "block");
        a.href = href; a.target = "_blank"; a.rel = "noopener noreferrer";
        const im = el("img", "h-24 w-auto rounded border border-gray-300");
        im.src = img.url; im.loading = "lazy"; im.alt = (img.doc_id || "") + " " + (img.ref || "");
        a.appendChild(im);
        box.appendChild(a);
      }
      card.appendChild(box);
    }
  }

  function citationItem(c) {
    const li = el("li", "flex items-start gap-1.5");
    li.appendChild(el("span", "text-gray-400 select-none", "↳"));
    const label = c.title || c.url || "(untitled source)";
    const href = safeHref(c.url);
    if (href) {
      const a = el("a", "text-blue-600 hover:text-blue-800 hover:underline break-all", label);
      a.href = href; a.target = "_blank"; a.rel = "noopener noreferrer";
      li.appendChild(a);
    } else {
      li.appendChild(el("span", "break-all", label));
    }
    return li;
  }

  // --- Turn driver ---

  async function runTurn(form) {
    const input = form.querySelector('input[name="message"]');
    const message = (input.value || "").trim();
    const sessionId = form.dataset.sessionId || "";
    if (!message) return;
    input.value = "";

    const messages = document.getElementById("chat-messages");
    messages.appendChild(userBubble(message));
    const a = assistantHost();
    messages.appendChild(a.wrap);
    scrollToBottom();

    const cards = new Map();
    let gotToken = false;

    try {
      const resp = await fetch("/v1/agent/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Session-ID": sessionId,
          "Accept": "text/event-stream",
        },
        body: JSON.stringify({ message: message, stream: true }),
      });
      if (!resp.ok || !resp.body) {
        throw new Error("HTTP " + resp.status);
      }

      for await (const { event, data } of sseEvents(resp)) {
        if (event === "token" && data) {
          if (!gotToken) { a.thinking.remove(); a.bubble.hidden = false; gotToken = true; }
          a.bubble.textContent += data.text || "";
          scrollToBottom();
        } else if (event === "tool" && data) {
          if (data.phase === "start") {
            const card = toolCardStart(data.call_id, data.name);
            cards.set(data.call_id, card);
            a.tools.appendChild(card);
          } else if (data.phase === "done") {
            const card = cards.get(data.call_id);
            if (card) toolCardDone(card, data);
          }
          scrollToBottom();
        } else if (event === "done") {
          break;
        } else if (event === "error" && data) {
          throw new Error(data.message || "stream error");
        }
      }
    } catch (err) {
      a.thinking.remove();
      a.bubble.hidden = false;
      a.bubble.classList.add("text-red-600");
      a.bubble.textContent = "[Error: " + (err && err.message ? err.message : String(err)) + "]";
    } finally {
      if (!gotToken && a.thinking.isConnected) a.thinking.remove();
      scrollToBottom();
    }
  }

  // Delegated submit — registered once, survives htmx-boost swaps of #main.
  document.addEventListener("submit", function (e) {
    const form = e.target;
    if (form && form.id === "chat-form") {
      e.preventDefault();
      runTurn(form);
    }
  });
})();
