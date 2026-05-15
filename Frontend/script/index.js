/* ═══════════════════════════════════════════════════
   Medical Detector — index.js
═══════════════════════════════════════════════════ */

const API = window.location.origin;

let imgFile   = null;
let connected = false;

const CLASSES = {
  rim:   { cor: "var(--rim)",   corBg: "var(--rim-bg)",   rotulo: "Rim",   desc: "Parênquima renal saudável" },
  tumor: { cor: "var(--tumor)", corBg: "var(--tumor-bg)", rotulo: "Tumor", desc: "Massa maligna"             },
  cisto: { cor: "var(--cisto)", corBg: "var(--cisto-bg)", rotulo: "Cisto", desc: "Cisto renal benigno"       },
};

/* ════════════════════════════════════════════════════
   HEALTH CHECK
════════════════════════════════════════════════════ */
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
    if (!r.ok) throw new Error();

    connected = true;
    const pill = document.getElementById("statusPill");
    pill.className = "status-pill online";
    document.getElementById("statusText").textContent = "Sistema online";
    if (imgFile) habilitarBtn();

  } catch {
    connected = false;
    const pill = document.getElementById("statusPill");
    pill.className = "status-pill error";
    document.getElementById("statusText").textContent = "Sem conexão";
    setTimeout(checkHealth, 3000);
  }
}

window.addEventListener("load", checkHealth);

/* ════════════════════════════════════════════════════
   UPLOAD — zona principal (placeholder)
════════════════════════════════════════════════════ */
const zone = document.getElementById("uploadZone");

zone.addEventListener("dragover", e => {
  e.preventDefault();
  zone.classList.add("drag");
});

zone.addEventListener("dragleave", e => {
  if (!zone.contains(e.relatedTarget)) zone.classList.remove("drag");
});

zone.addEventListener("drop", e => {
  e.preventDefault();
  zone.classList.remove("drag");
  const f = e.dataTransfer.files[0];
  if (f) processFile(f);
});

zone.addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    document.getElementById("fileInput").click();
  }
});

function onFile(input) {
  if (input.files[0]) processFile(input.files[0]);
}

/* ════════════════════════════════════════════════════
   PROCESSA ARQUIVO
════════════════════════════════════════════════════ */
function processFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  const extOk = ["png","jpg","jpeg","tif","tiff","dcm","dicom"].includes(ext);

  if (!file.type.startsWith("image/") && !extOk) {
    toast("Formato não suportado. Use PNG, JPG, TIFF ou DICOM.");
    return;
  }

  // Libera URL anterior
  const previewEl = document.getElementById("previewImg");
  if (previewEl.src && previewEl.src.startsWith("blob:")) {
    URL.revokeObjectURL(previewEl.src);
  }

  imgFile = file;

  // Atualiza preview
  previewEl.src = URL.createObjectURL(file);
  previewEl.alt = `Imagem carregada: ${file.name}`;

  // Ativa modo com imagem
  zone.classList.add("has-image");

  // Mostra info do arquivo
  const fileInfo = document.getElementById("fileInfo");
  fileInfo.style.display = "flex";
  document.getElementById("fileInfoText").textContent =
    `${file.name} · ${formatBytes(file.size)}`;

  // Limpa resultados anteriores
  limparResultados();

  if (connected) habilitarBtn();
}

function limparResultados() {
  document.getElementById("resultsList").innerHTML = `
    <div class="empty-state">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="1" aria-hidden="true">
        <circle cx="11" cy="11" r="8"/>
        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
        <line x1="11" y1="8" x2="11" y2="14"/>
        <line x1="8" y1="11" x2="14" y2="11"/>
      </svg>
      <p>Imagem carregada.<br/>Clique em <strong>Analisar imagem</strong>.</p>
    </div>`;
  document.getElementById("badgeTotal").textContent = "—";
  document.getElementById("summaryPanel").style.display = "none";
}

/* ════════════════════════════════════════════════════
   UTILITÁRIOS
════════════════════════════════════════════════════ */
function formatBytes(bytes) {
  if (bytes < 1024)        return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function updateSlider(sliderId, labelId, value) {
  document.getElementById(labelId).textContent = value;
  document.getElementById(sliderId).setAttribute("aria-valuenow",
    document.getElementById(sliderId).value);
}

function habilitarBtn() {
  document.getElementById("btnPredict").disabled = false;
  document.getElementById("btnLabel").textContent = "Analisar imagem";
}

/* ════════════════════════════════════════════════════
   PREDIÇÃO
════════════════════════════════════════════════════ */
async function predict() {
  if (!imgFile || !connected) return;

  setLoading(true);

  const conf = document.getElementById("confSlider").value / 100;
  const iou  = document.getElementById("iouSlider").value  / 100;

  const form = new FormData();
  form.append("file", imgFile);

  try {
    const r = await fetch(`${API}/predict?conf=${conf}&iou=${iou}`, {
      method: "POST", body: form,
    });

    if (!r.ok) throw new Error(`Erro ${r.status}: ${await r.text()}`);

    renderResultados(await r.json());

  } catch (e) {
    const msg = e.message.includes("Failed to fetch")
      ? "Não foi possível conectar ao servidor."
      : e.message;
    toast(msg);

  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  const btn     = document.getElementById("btnPredict");
  const spinner = document.getElementById("spinner");
  const label   = document.getElementById("btnLabel");

  spinner.style.display = on ? "block" : "none";
  label.textContent     = on ? "Analisando..." : "Analisar novamente";
  btn.disabled          = on;
  on ? btn.setAttribute("aria-busy","true") : btn.removeAttribute("aria-busy");
}

/* ════════════════════════════════════════════════════
   RENDERIZAÇÃO
════════════════════════════════════════════════════ */
function renderResultados(data) {
  // Atualiza preview com imagem anotada
  const img = document.getElementById("previewImg");
  img.src = data.imagem_anotada;
  img.alt = `Resultado: ${data.total_deteccoes} detecção(ões)`;

  // Badge
  const total = data.total_deteccoes;
  document.getElementById("badgeTotal").textContent =
    total + (total === 1 ? " detecção" : " detecções");

  // Lista
  const list = document.getElementById("resultsList");
  list.innerHTML = "";

  if (total === 0) {
    const conf = document.getElementById("confSlider").value;
    list.innerHTML = `
      <div class="empty-state">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="1" aria-hidden="true">
          <circle cx="12" cy="12" r="10"/>
          <line x1="15" y1="9" x2="9" y2="15"/>
          <line x1="9" y1="9" x2="15" y2="15"/>
        </svg>
        <p>Nenhuma detecção acima de <strong>${conf}%</strong>.<br/>
           Tente reduzir a confiança mínima.</p>
      </div>`;
  } else {
    data.deteccoes.forEach((d, i) => {
      const cls = CLASSES[d.classe] || {
        cor: "var(--primary)", corBg: "var(--primary-light)",
        rotulo: d.classe, desc: "",
      };
      const pct  = Math.round(d.confianca * 100);
      const b    = d.bbox || {};
      const card = document.createElement("div");
      card.className = "det-card";
      card.setAttribute("role", "listitem");
      card.style.cssText = `animation-delay:${i*60}ms; border-left:3px solid ${cls.cor}`;

      card.innerHTML = `
        <div class="class-dot" style="background:${cls.cor}" aria-hidden="true"></div>
        <div class="det-info">
          <div class="det-name" style="color:${cls.cor}">
            ${cls.rotulo}
            <span style="font-size:10px;font-weight:500;color:var(--muted);margin-left:6px">
              ${cls.desc}
            </span>
          </div>
          <div class="det-meta">
            x1:${b.x1??0} y1:${b.y1??0} → x2:${b.x2??0} y2:${b.y2??0}
            &nbsp;·&nbsp; ${(d.area_px??0).toLocaleString("pt-BR")} px²
          </div>
        </div>
        <div class="det-conf-col">
          <div class="conf-num" style="color:${cls.cor}">${pct}%</div>
          <div class="conf-lbl">confiança</div>
        </div>
        <div class="conf-bar-row" aria-hidden="true">
          <div class="conf-bar" style="width:${pct}%;background:${cls.cor}"></div>
        </div>`;

      list.appendChild(card);
    });
  }

  // Sumário
  const count = { rim:0, tumor:0, cisto:0 };
  data.deteccoes.forEach(d => { if (d.classe in count) count[d.classe]++; });
  document.getElementById("sumRim").textContent   = count.rim;
  document.getElementById("sumTumor").textContent = count.tumor;
  document.getElementById("sumCisto").textContent = count.cisto;
  document.getElementById("summaryPanel").style.display = "block";
}

/* ════════════════════════════════════════════════════
   TOAST
════════════════════════════════════════════════════ */
let toastTimer = null;

function toast(msg) {
  const t = document.getElementById("toast");
  document.getElementById("toastMsg").textContent = msg;
  t.classList.add("show");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 5000);
}