"use strict";
const $ = (id) => document.getElementById(id);
// /dev is the team page: key login, shared queue and history even when the site runs in guest mode.
const DEV = location.pathname.replace(/\/+$/, "") === "/dev";
if (DEV) document.querySelector(".brand").href = "/dev";
const S = {
  key: "",
  connected: false,
  job: null,
  file: null,
  result: null,
  generation: 0,
  frameGeneration: 0,
  objects: null,
  videoURL: null,
};
const stageNames = {
  queued: "Видео в очереди",
  frames: "Подготовка кадров",
  objects: "Объекты и треки",
  zones: "Определение зон",
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
  if (S.key) headers.set("X-API-Key", S.key);
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
// Guests have no key form, so the panel shows only the message.
function notify(message) {
  $("connection-status").textContent = message;
  if (S.guest) {
    $("connection-panel").classList.add("notice-only");
    show("connection-panel");
  }
}
async function json(path, options) {
  return (await api(path, options)).json();
}
async function history() {
  if (!S.connected || S.guest) return;
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
    S.connected = true;
    $("connection-status").textContent = "Подключено";
    await history();
    await benchmarkAvailable();
  } catch (e) {
    $("connection-status").textContent = e.message;
  }
};
$("api-key").onkeydown = (e) => {
  if (e.key === "Enter") $("connect").click();
};
function navigate(view) {
  for (const id of ["upload-view", "job-view", "benchmark-view"])
    show(id, id === view);
  $("new-job").classList.toggle("active", view !== "benchmark-view");
  $("benchmark-nav").classList.toggle("active", view === "benchmark-view");
}
$("new-job").onclick = () => {
  S.generation++;
  S.job = null;
  S.result = null;
  navigate("upload-view");
  if (S.videoURL) {
    URL.revokeObjectURL(S.videoURL);
    S.videoURL = null;
  }
};
function chooseFile(file) {
  if (!file) return;
  S.file = file;
  $("file-title").textContent = file.name;
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
$("upload-form").onsubmit = async (e) => {
  e.preventDefault();
  if (!S.connected) S.key = $("api-key").value.trim();
  if (!S.connected && !S.key) {
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
  $("analyze-button").disabled = true;
  try {
    const d = await json("/api/v1/analyze", { method: "POST", body });
    await selectJob(d.job_id, file.name);
  } catch (e) {
    notify(e.message);
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
  navigate("job-view");
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
async function renderSummary(result, id) {
  show("summary-intro", false);
  show("summary-observed-card", false);
  let summary;
  try {
    summary = await json(`/api/v1/jobs/${id}/artifacts/summary`);
  } catch {
    return; // Interface-only artifact; a missing/failed fetch shows no error, just no extra block.
  }
  const noViolation = result.verdict !== "violation";
  const observed = summary.observed || [];
  if (noViolation) {
    show("summary-intro", true);
    if (observed.length) {
      show("summary-observed-card", true);
      $("summary-observed").replaceChildren(...observed.map((text) => el("li", text)));
    }
  }
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
  await renderSummary(result, id);
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
        ctx.strokeStyle = d.label === "person" ? "#00c8f0" : "#f4917f";
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
json("/api/v1/access")
  .then(async (access) => {
    if (access.guest && !DEV) {
      // Public site (guest mode): no key and no history list; the shared queue is for the team copy.
      S.guest = true;
      S.connected = true;
      show("history-section", false);
    } else if (access.authorized) {
      S.connected = true;
      await history();
      await benchmarkAvailable();
    } else {
      show("connection-panel");
    }
  })
  .catch((error) => {
    show("connection-panel");
    $("connection-status").textContent = error.message;
  });

// Benchmark: labels and comments are rendered only here, never sent anywhere else.
const outcomeNames = {
  TP: "Нашли",
  FN: "Пропуск",
  FP: "Ложное",
  TN: "Верно: нет",
  failed: "Ошибка",
};
const runStatusNames = {
  running: "Выполняется",
  cancelling: "Останавливается",
  done: "Завершён",
  cancelled: "Остановлен",
  interrupted: "Прерван перезапуском",
};
function percent(value) {
  return value === null || value === undefined
    ? "—"
    : Math.round(value * 100) + "%";
}
async function benchmarkAvailable() {
  try {
    const d = await json("/api/v1/benchmarks");
    show("benchmark-nav", d.enabled);
  } catch {
    show("benchmark-nav", false);
  }
}
async function loadBenchmarks() {
  clearTimeout(S.benchmarkTimer);
  const d = await json("/api/v1/benchmarks");
  const running = d.runs.find((r) => r.run_id === d.running);
  $("benchmark-run").disabled = Boolean(d.running);
  show("benchmark-cancel", Boolean(running && running.status === "running"));
  S.benchmarkRunning = d.running;
  const latest = running || d.runs[0];
  if (latest) {
    const s = latest.summary;
    $("benchmark-status").textContent =
      (runStatusNames[latest.status] || latest.status) +
      " · обработано " +
      s.processed +
      " из " +
      s.videos +
      (latest.status === "running"
        ? " · загрузки пользователей обрабатываются между видео"
        : "");
    $("benchmark-metrics").replaceChildren(
      ...[
        [percent(s.accuracy), "Верный вердикт"],
        [percent(s.recall), `Найдено нарушений (${s.TP} из ${s.TP + s.FN})`],
        [s.FP, "Ложных срабатываний"],
        [
          s.located_of ? `${s.located}/${s.located_of}` : "—",
          `Таймкод ±${latest.config.tolerance_sec} с`,
        ],
        [
          s.judged ? `${s.judge_yes}/${s.judged}` : "—",
          "Судья: то же нарушение",
        ],
        [s.judged ? `${s.judge_type_match}/${s.judged}` : "—", "Судья: верный тип"],
        [
          s.max_processing_sec === null ? "—" : s.max_processing_sec + " с",
          "Макс. время на видео",
        ],
        [s.failed + s.over_time_limit, "Сбоев и превышений лимита"],
      ].map(([value, title]) => {
        const n = el("div", null, "metric");
        n.append(el("strong", value), el("span", title));
        return n;
      }),
    );
  } else {
    $("benchmark-status").textContent =
      "Прогонов ещё не было. Нажмите Run benchmark.";
    $("benchmark-metrics").replaceChildren();
  }
  $("benchmark-runs").replaceChildren(
    table(
      [
        "Запуск",
        "Статус",
        "Версия",
        "Модели",
        "Вердикт",
        "Recall",
        "Ложные",
        "Таймкод",
        "Судья",
      ],
      d.runs.map((r) => {
        const s = r.summary,
          open = el("button", new Date(r.started_at * 1000).toLocaleString());
        open.onclick = () => showBenchmark(r.run_id);
        return [
          open,
          (runStatusNames[r.status] || r.status) + ` (${s.processed}/${s.videos})`,
          (r.config.release || "").slice(0, 7),
          [r.config.model, r.config.llm_model || "без LLM"].join(" · "),
          percent(s.accuracy),
          percent(s.recall),
          s.FP,
          s.located_of ? `${s.located}/${s.located_of}` : "—",
          s.judged ? `${s.judge_yes}/${s.judged}` : "—",
        ];
      }),
    ),
  );
  const selected = S.benchmarkSelected || latest?.run_id;
  if (selected) await showBenchmark(selected);
  if (d.running && !$("benchmark-view").classList.contains("hidden"))
    S.benchmarkTimer = setTimeout(() => loadBenchmarks().catch(() => {}), 5000);
}
async function showBenchmark(runId) {
  S.benchmarkSelected = runId;
  const run = await json("/api/v1/benchmarks/" + runId);
  show("benchmark-detail");
  $("benchmark-detail-title").textContent =
    "Прогон " + new Date(run.started_at * 1000).toLocaleString();
  const c = run.config;
  $("benchmark-detail-config").textContent = [
    "версия " + (c.release || "").slice(0, 7),
    c.detector + " · " + c.model,
    c.sample_fps + " fps",
    "LLM " + (c.llm_model || "выкл."),
    "судья " + c.judge_model,
    "разметка " + c.labels_sha,
  ].join(" · ");
  $("benchmark-warnings").replaceChildren(
    ...run.warnings.map((w) => el("li", w)),
  );
  $("benchmark-items").replaceChildren(
    table(
      [
        "Видео",
        "Разметка",
        "Ответ сервиса",
        "Итог",
        "Ошибка таймкода",
        "Судья",
        "Комментарий разметки",
      ],
      run.items.map((item) => {
        const label = item.label,
          s = item.score || {},
          expected = label.violation
            ? "Нарушение" +
              (label.times.length
                ? " · " + label.times.map((t) => clock(t)).join(", ")
                : "")
            : "Без нарушения";
        let answer = "—";
        if (item.status === "running") answer = "Обрабатывается…";
        else if (item.status === "pending") answer = "В очереди";
        else if (item.status === "skipped") answer = "Пропущено";
        else if (item.error) answer = "Ошибка: " + item.error;
        else if (item.violations)
          answer = item.violations.length
            ? item.violations
                .map((v) => clock(v.timecode_sec) + " " + v.type)
                .join("; ")
            : "Без нарушения";
        const outcome = el(
          "span",
          outcomeNames[s.outcome] || "—",
          "outcome outcome-" + (s.outcome || "none"),
        );
        let judge = "—";
        if (typeof item.judge === "string") judge = item.judge;
        else if (item.judge)
          judge =
            { yes: "Да", partial: "Частично", no: "Нет" }[item.judge.same_event] +
            (item.judge.type_match ? ", тип верный" : ", тип неверный") +
            " — " +
            item.judge.explanation;
        return [
          item.video,
          expected,
          answer,
          outcome,
          s.time_error_sec === null || s.time_error_sec === undefined
            ? "—"
            : s.time_error_sec + " с",
          judge,
          label.comment || "—",
        ];
      }),
    ),
  );
}
$("benchmark-nav").onclick = () => {
  navigate("benchmark-view");
  loadBenchmarks().catch((e) => {
    $("benchmark-status").textContent = e.message;
  });
};
$("benchmark-run").onclick = async () => {
  if (
    !confirm(
      "Прогнать все размеченные видео? Это займёт десятки минут. Не запускайте во время проверки жюри.",
    )
  )
    return;
  try {
    const r = await json("/api/v1/benchmarks", { method: "POST" });
    S.benchmarkSelected = r.run_id;
    await loadBenchmarks();
  } catch (e) {
    $("benchmark-status").textContent = e.message;
  }
};
$("benchmark-cancel").onclick = async () => {
  if (!S.benchmarkRunning) return;
  await json(`/api/v1/benchmarks/${S.benchmarkRunning}/cancel`, {
    method: "POST",
  });
  await loadBenchmarks();
};
