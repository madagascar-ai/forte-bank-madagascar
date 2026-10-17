"use strict";
const $ = (id) => document.getElementById(id);
const S = {
  key: "",
  job: null,
  file: null,
  result: null,
  zones: [],
  generation: 0,
  frameGeneration: 0,
  objects: null,
  videoURL: null,
  previewURL: null,
};
const stageNames = {
  queued: "Видео в очереди",
  frames: "Подготовка кадров",
  zones: "Определение зон",
  objects: "Объекты и треки",
  actions: "Журнал действий",
  candidates: "Правила + LLM",
  validation: "Сборка результата",
};
const actionNames = {
  client_observed: "Клиент наблюдается у кассы",
  cash_transferred: "Передача денег",
  cashier_left_workstation: "Кассир покинул рабочее место",
  cashier_returned: "Кассир вернулся",
  cash_placed_on_counter: "Деньги положены на стол",
  device_used_observed: "Взаимодействие с устройством",
  document_presented: "Действие с документом",
  cash_entered_storage_area: "Перемещение денег в зону хранения",
};
const colors = {
  cashier_area: "#a82e64",
  client_area: "#3f8c92",
  counter_area: "#dd9f42",
  storage_area: "#7662be",
  exit_area: "#5d8d46",
};
function el(tag, text, cls) {
  const n = document.createElement(tag);
  if (text !== undefined && text !== null) n.textContent = String(text);
  if (cls) n.className = cls;
  return n;
}
function show(id, visible = true) {
  $(id).classList.toggle("hidden", !visible);
}
function clock(seconds) {
  return (
    String(Math.floor(seconds / 60)).padStart(2, "0") +
    ":" +
    (seconds % 60).toFixed(2).padStart(5, "0")
  );
}
async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-API-Key", S.key);
  const r = await fetch(path, { ...options, headers });
  if (!r.ok) {
    let message = "Ошибка " + r.status;
    try {
      const d = await r.json();
      if (typeof d.detail === "string") message = d.detail;
    } catch {}
    throw new Error(message);
  }
  return r;
}
async function json(path, options) {
  return (await api(path, options)).json();
}
async function history() {
  if (!S.key) return;
  const d = await json("/api/v1/jobs");
  $("history").replaceChildren(
    ...d.jobs.map((j) => {
      const b = el("button", j.filename);
      b.append(
        el(
          "small",
          j.status === "done"
            ? "Обработано"
            : j.status === "failed"
              ? "Ошибка"
              : "Обрабатывается",
        ),
      );
      b.onclick = () => selectJob(j.job_id, j.filename);
      return b;
    }),
  );
}
$("connect").onclick = async () => {
  S.key = $("api-key").value.trim();
  try {
    await json("/api/v1/config");
    $("connection-status").textContent = "Подключено";
    await history();
  } catch (e) {
    $("connection-status").textContent = e.message;
  }
};
$("api-key").onkeydown = (e) => {
  if (e.key === "Enter") $("connect").click();
};
$("new-job").onclick = () => {
  S.generation++;
  S.job = null;
  S.result = null;
  show("upload-view");
  show("job-view", false);
  if (S.videoURL) {
    URL.revokeObjectURL(S.videoURL);
    S.videoURL = null;
  }
};
function chooseFile(file) {
  if (!file) return;
  S.file = file;
  S.zones = [];
  $("file-title").textContent = file.name;
  if (S.previewURL) URL.revokeObjectURL(S.previewURL);
  S.previewURL = URL.createObjectURL(file);
  const v = $("preview-video");
  v.src = S.previewURL;
  v.onloadeddata = () => {
    const c = $("zone-canvas");
    c.width = Math.min(640, v.videoWidth);
    c.height = Math.round((c.width * v.videoHeight) / v.videoWidth);
    show("zone-panel");
    drawZones();
  };
  v.onerror = () => show("zone-panel", false);
  v.load();
  $("zone-count").textContent = "Автоматическое определение";
}
$("video-file").onchange = (e) => chooseFile(e.target.files[0]);
["dragenter", "dragover", "dragleave", "drop"].forEach((name) =>
  $("dropzone").addEventListener(name, (e) => {
    e.preventDefault();
    $("dropzone").classList.toggle(
      "drag",
      ["dragenter", "dragover"].includes(name),
    );
    if (name === "drop") chooseFile(e.dataTransfer.files[0]);
  }),
);
let dragging = null;
function point(e) {
  const r = $("zone-canvas").getBoundingClientRect();
  return [
    Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
    Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)),
  ];
}
function rectangle(a, b) {
  return {
    x1: Math.min(a[0], b[0]),
    y1: Math.min(a[1], b[1]),
    x2: Math.max(a[0], b[0]),
    y2: Math.max(a[1], b[1]),
  };
}
function drawZones(preview) {
  const c = $("zone-canvas"),
    ctx = c.getContext("2d"),
    v = $("preview-video");
  if (v.readyState < 2) return;
  ctx.drawImage(v, 0, 0, c.width, c.height);
  for (const z of [...S.zones, ...(preview ? [preview] : [])]) {
    const b = z.box;
    ctx.strokeStyle = colors[z.name];
    ctx.fillStyle = colors[z.name] + "26";
    ctx.lineWidth = 2;
    ctx.fillRect(
      b.x1 * c.width,
      b.y1 * c.height,
      (b.x2 - b.x1) * c.width,
      (b.y2 - b.y1) * c.height,
    );
    ctx.strokeRect(
      b.x1 * c.width,
      b.y1 * c.height,
      (b.x2 - b.x1) * c.width,
      (b.y2 - b.y1) * c.height,
    );
    ctx.fillStyle = colors[z.name];
    ctx.font = "12px sans-serif";
    ctx.fillText(z.name, b.x1 * c.width + 4, b.y1 * c.height + 15);
  }
}
$("zone-canvas").onpointerdown = (e) => {
  dragging = point(e);
  $("zone-canvas").setPointerCapture(e.pointerId);
};
$("zone-canvas").onpointermove = (e) => {
  if (dragging)
    drawZones({
      name: $("zone-type").value,
      box: rectangle(dragging, point(e)),
    });
};
$("zone-canvas").onpointerup = (e) => {
  if (!dragging) return;
  const name = $("zone-type").value,
    box = rectangle(dragging, point(e));
  dragging = null;
  if (box.x2 - box.x1 > 0.02 && box.y2 - box.y1 > 0.02) {
    S.zones = S.zones.filter((z) => z.name !== name);
    S.zones.push({ name, box });
  }
  $("zone-count").textContent = S.zones.length
    ? "Задано зон: " + S.zones.length
    : "Автоматическое определение";
  drawZones();
};
$("clear-zones").onclick = () => {
  S.zones = [];
  $("zone-count").textContent = "Автоматическое определение";
  drawZones();
};
$("upload-form").onsubmit = async (e) => {
  e.preventDefault();
  S.key = $("api-key").value.trim();
  if (!S.key) {
    $("connection-status").textContent = "Введите ключ доступа";
    $("api-key").focus();
    return;
  }
  const file = S.file || $("video-file").files[0];
  if (!file) {
    $("file-title").textContent = "Выберите файл";
    return;
  }
  const body = new FormData();
  body.append("file", file);
  if (S.zones.length)
    body.append("zones", JSON.stringify({ source: "manual", zones: S.zones }));
  $("analyze-button").disabled = true;
  try {
    const d = await json("/api/v1/analyze", { method: "POST", body });
    await selectJob(d.job_id, file.name);
  } catch (e) {
    $("connection-status").textContent = e.message;
  } finally {
    $("analyze-button").disabled = false;
  }
};
function progress(j) {
  const p = Math.round((j.progress || 0) * 100);
  $("stage-label").textContent = stageNames[j.stage] || "Обработка";
  $("progress-label").textContent = p + "%";
  $("progress-bar").style.width = p + "%";
  const keys = Object.keys(stageNames).filter((k) => k !== "queued"),
    i = keys.indexOf(j.stage);
  $("pipeline-stages").replaceChildren(
    ...keys.map((k, n) =>
      el(
        "span",
        (n < i ? "✓ " : "") + stageNames[k],
        n < i ? "done" : n === i ? "current" : "",
      ),
    ),
  );
}
async function selectJob(id, name) {
  const generation = ++S.generation;
  S.job = id;
  S.result = null;
  S.objects = null;
  S.frameGeneration++;
  if (S.videoURL) {
    URL.revokeObjectURL(S.videoURL);
    S.videoURL = null;
  }
  show("upload-view", false);
  show("job-view");
  show("processing");
  show("result-panel", false);
  show("error-box", false);
  show("download-json", false);
  $("job-title").textContent = name;
  $("job-subtitle").textContent = "Проверка " + id.slice(0, 8);
  progress({ stage: "queued", progress: 0 });
  async function poll() {
    try {
      const j = await json("/api/v1/jobs/" + id);
      if (generation !== S.generation) return;
      if (j.status === "done") {
        S.result = j;
        show("processing", false);
        show("result-panel");
        show("download-json");
        await renderResult(j, generation, id);
        await history();
      } else if (j.status === "failed") {
        show("processing", false);
        show("error-box");
        $("error-box").textContent = j.error || "Ошибка обработки";
        await history();
      } else {
        progress(j);
        setTimeout(poll, 1200);
      }
    } catch (e) {
      if (generation === S.generation) {
        show("processing", false);
        show("error-box");
        $("error-box").textContent = e.message;
      }
    }
  }
  await poll();
}
function seek(seconds) {
  activateTab("findings");
  const p = $("source-player");
  if (p) {
    p.currentTime = seconds;
    p.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}
function table(headers, rows) {
  const t = el("table"),
    head = el("thead"),
    hr = el("tr");
  headers.forEach((h) => hr.append(el("th", h)));
  head.append(hr);
  t.append(head);
  const body = el("tbody");
  for (const row of rows) {
    const tr = el("tr");
    for (const cell of row) {
      const td = el("td");
      if (cell instanceof Node) td.append(cell);
      else td.textContent = String(cell ?? "");
      tr.append(td);
    }
    body.append(tr);
  }
  t.append(body);
  return t;
}
async function renderResult(result, generation, id) {
  const positive = result.verdict === "violation";
  $("verdict").className = "verdict" + (positive ? "" : " clear");
  $("verdict").replaceChildren(
    el(
      "div",
      positive
        ? "Обнаружены признаки возможных нарушений"
        : "Признаки нарушений не выявлены",
    ),
    el(
      "small",
      positive
        ? "Каждое событие требует проверки по записи и применимым правилам."
        : "Результат относится к доступным наблюдениям и включённым правилам.",
    ),
  );
  const s = result.stats || {},
    metrics = [
      [s.frames || 0, "Проанализировано кадров"],
      [s.actions || 0, "Событий в журнале"],
      [s.merged_candidates || 0, "Кандидатов"],
      [result.violations.length, "Событий для проверки"],
    ];
  $("metrics").replaceChildren(
    ...metrics.map(([value, title]) => {
      const n = el("div", null, "metric");
      n.append(el("strong", value), el("span", title));
      return n;
    }),
  );
  $("findings").replaceChildren();
  const videoCard = el("div", null, "card"),
    player = el("video");
  player.id = "source-player";
  player.controls = true;
  player.preload = "metadata";
  videoCard.append(el("h3", "Исходная запись"), player);
  $("findings").append(videoCard);
  if (!result.violations.length)
    $("findings").append(
      el(
        "div",
        "Событий для проверки не найдено. Откройте журнал действий и объекты для просмотра наблюдений.",
        "empty",
      ),
    );
  for (const finding of result.violations) {
    const card = el("article", null, "finding"),
      top = el("div", null, "finding-top"),
      button = el("button", clock(finding.timecode_sec), "timestamp");
    button.onclick = () => seek(finding.timecode_sec);
    top.append(button, el("h3", finding.type));
    card.append(top, el("p", finding.description));
    const evidence = el("div", null, "evidence");
    for (const frame of finding.evidence_frames) {
      const f = el("figure"),
        img = el("img");
      img.src = "data:image/jpeg;base64," + frame.image_base64;
      img.alt = "Кадр-доказательство";
      f.append(img, el("figcaption", clock(frame.timestamp_sec)));
      evidence.append(f);
    }
    card.append(evidence);
    const source = el(
      "p",
      "Источники кандидата: " + finding.sources.join(", "),
      "source",
    );
    if (finding.document?.url) {
      try {
        const url = new URL(finding.document.url);
        if (["http:", "https:"].includes(url.protocol)) {
          const a = el("a", "Нормативный источник");
          a.href = url.href;
          a.target = "_blank";
          a.rel = "noopener noreferrer";
          source.append(document.createTextNode(" · "), a);
        }
      } catch {}
    }
    card.append(source);
    $("findings").append(card);
  }
  $("warnings").replaceChildren(...result.warnings.map((w) => el("li", w)));
  const [actions, objects, candidates] = await Promise.all(
    ["action_journal", "object_journal", "candidates"].map((name) =>
      json(`/api/v1/jobs/${id}/artifacts/${name}`),
    ),
  );
  if (generation !== S.generation) return;
  S.objects = objects;
  $("action-count").textContent = actions.actions.length + " событий";
  $("actions-table").replaceChildren(
    table(
      ["Время", "Действие", "Участники", "Объекты", "Видимость"],
      actions.actions.map((a) => {
        const time = el("button", clock(a.start_sec));
        time.onclick = () => seek(a.start_sec);
        return [
          time,
          actionNames[a.type] || a.type,
          a.actor_ids.join(", "),
          a.object_ids.join(", "),
          a.visibility === "clear" ? "Наблюдается" : "Неуверенно",
        ];
      }),
    ),
  );
  $("states-json").textContent = JSON.stringify(
    {
      roles: actions.roles,
      batches: actions.batches,
      states: actions.states,
      unsupported_actions: actions.unsupported_actions,
      coverage: actions.coverage,
    },
    null,
    2,
  );
  $("candidates-json").textContent = JSON.stringify(candidates, null, 2);
  $("object-count").textContent =
    objects.frames.reduce((total, f) => total + f.detections.length, 0) +
    " обнаружений";
  $("frame-slider").max = Math.max(0, objects.frames.length - 1);
  $("frame-slider").value = 0;
  await renderFrame(0);
  activateTab("findings");
  const blob = await (await api(`/api/v1/jobs/${id}/video`)).blob();
  if (generation !== S.generation) return;
  S.videoURL = URL.createObjectURL(blob);
  player.src = S.videoURL;
}
async function renderFrame(index) {
  if (!S.objects?.frames[index]) return;
  const generation = ++S.frameGeneration,
    id = S.job,
    f = S.objects.frames[index];
  $("frame-label").textContent =
    clock(f.timestamp_sec) +
    " · " +
    f.frame_id +
    " · объектов: " +
    f.detections.length;
  $("objects-table").replaceChildren(
    table(
      ["ID", "Объект", "Уверенность", "Состояние"],
      f.detections.map((d) => [
        d.track_id,
        d.label,
        d.confidence.toFixed(2),
        d.attributes.state || "—",
      ]),
    ),
  );
  const blob = await (
      await api(`/api/v1/jobs/${id}/frames/${f.frame_id}`)
    ).blob(),
    url = URL.createObjectURL(blob),
    image = new Image();
  image.onload = () => {
    if (generation === S.frameGeneration && id === S.job) {
      const c = $("object-canvas");
      c.width = image.width;
      c.height = image.height;
      const ctx = c.getContext("2d");
      ctx.drawImage(image, 0, 0);
      ctx.lineWidth = 2;
      ctx.font = "12px sans-serif";
      for (const d of f.detections) {
        const b = d.box;
        ctx.strokeStyle = d.label === "person" ? "#42d9a2" : "#ffba67";
        ctx.fillStyle = ctx.strokeStyle;
        ctx.strokeRect(
          b.x1 * c.width,
          b.y1 * c.height,
          (b.x2 - b.x1) * c.width,
          (b.y2 - b.y1) * c.height,
        );
        ctx.fillText(
          d.label + " " + d.track_id,
          b.x1 * c.width,
          Math.max(12, b.y1 * c.height - 4),
        );
      }
    }
    URL.revokeObjectURL(url);
  };
  image.onerror = () => URL.revokeObjectURL(url);
  image.src = url;
}
let sliderTimer;
$("frame-slider").oninput = (e) => {
  clearTimeout(sliderTimer);
  sliderTimer = setTimeout(
    () =>
      renderFrame(Number(e.target.value)).catch((e) => {
        $("frame-label").textContent = e.message;
      }),
    120,
  );
};
function activateTab(name) {
  document
    .querySelectorAll(".tab")
    .forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  for (const n of ["findings", "actions", "objects", "candidates"])
    show("tab-" + n, n === name);
}
document
  .querySelectorAll(".tab")
  .forEach((b) => (b.onclick = () => activateTab(b.dataset.tab)));
$("download-json").onclick = () => {
  if (!S.result) return;
  const url = URL.createObjectURL(
      new Blob([JSON.stringify(S.result, null, 2)], {
        type: "application/json",
      }),
    ),
    a = el("a");
  a.href = url;
  a.download = "forte-" + S.job + ".json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
fetch("/health")
  .then((r) => r.json())
  .then((d) => {
    $("server-status").textContent = d.detector_ready
      ? "Локальный сервис готов"
      : "Сервис запущен · подготовьте детектор";
    if (/^[0-9a-f]{40}$/.test(d.release || "")) {
      $("release-info").textContent =
        "Версия " + d.release.slice(0, 7) + " · Локальная обработка";
    }
  })
  .catch(() => {
    $("server-status").textContent = "Сервис недоступен";
  });
