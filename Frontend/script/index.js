/* ═══════════════════════════════════════════════════
   YOLO Medical Detector — index.js
   Responsabilidades:
   - Conexão com backend (health check com retry)
   - Upload via clique e drag & drop
   - Predição com feedback de loading
   - Renderização de cards com cores semânticas
   - Toast de erro com acessibilidade
═══════════════════════════════════════════════════ */

// ── API — mesmo servidor que serve o HTML ─────────────────────────────────────
const API = window.location.origin;

// ── Estado global ─────────────────────────────────────────────────────────────
let imgFile   = null;
let connected = false;

// ── Cores e rótulos semânticos por classe ─────────────────────────────────────
// Alinhados com o CSS (--rim, --tumor, --cisto)
const CLASSES = {
  rim: {
    cor:    "var(--rim)",
    corBg:  "var(--rim-bg)",
    rotulo: "Rim",
    desc:   "Parênquima renal saudável",
  },
  tumor: {
    cor:    "var(--tumor)",
    corBg:  "var(--tumor-bg)",
    rotulo: "Tumor",
    desc:   "Massa maligna",
  },
  cisto: {
    cor:    "var(--cisto)",
    corBg:  "var(--cisto-bg)",
    rotulo: "Cisto",
    desc:   "Cisto renal benigno",
  },
};

// ════════════════════════════════════════════════════
// HEALTH CHECK — verifica conexão com o backend
// ════════════════════════════════════════════════════
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`, {
      signal: AbortSignal.timeout(4000),
    });

    if (!r.ok) throw new Error("Backend retornou erro");

    // Conectado com sucesso
    connected = true;

    const pill = document.getElementById("statusPill");
    pill.className = "status-pill online";
    document.getElementById("statusText").textContent = "Sistema online";

    if (imgFile) habilitarBtn();

  } catch {
    // Falha na conexão
    connected = false;

    const pill = document.getElementById("statusPill");
    pill.className = "status-pill error";
    document.getElementById("statusText").textContent = "Sem conexão";

    // Tenta reconectar após 3 segundos
    setTimeout(checkHealth, 3000);
  }
}

window.addEventListener("load", checkHealth);

// ════════════════════════════════════════════════════
// UPLOAD — clique e drag & drop
// ════════════════════════════════════════════════════
const zone = document.getElementById("uploadZone");

zone.addEventListener("dragover", e => {
  e.preventDefault();
  zone.classList.add("drag");
});

zone.addEventListener("dragleave", e => {
  // Ignora se o foco saiu para um filho
  if (!zone.contains(e.relatedTarget)) {
    zone.classList.remove("drag");
  }
});

zone.addEventListener("drop", e => {
  e.preventDefault();
  zone.classList.remove("drag");

  const f = e.dataTransfer.files[0];
  if (f) processFile(f);
});

// Acessibilidade: Enter/Space abrem o seletor de arquivo
zone.addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    document.getElementById("fileInput").click();
  }
});

function onFile(input) {
  if (input.files[0]) processFile(input.files[0]);
}

function processFile(file) {
  // Valida tipo de arquivo
  const tiposAceitos = ["image/png", "image/jpeg", "image/tiff",
                        "image/tif", "application/dicom", ""];
  const ext = file.name.split(".").pop().toLowerCase();
  const extAceita = ["png","jpg","jpeg","tif","tiff","dcm","dicom"].includes(ext);

  if (!file.type.startsWith("image/") && !extAceita) {
    toast("Formato não suportado. Use PNG, JPG, TIFF ou DICOM.");
    return;
  }

  imgFile = file;

  // Mostra preview
  const url = URL.createObjectURL(file);
  document.getElementById("previewImg").src = url;
  document.getElementById("previewImg").alt =
    `Imagem carregada: ${file.name}`;

  document.getElementById("previewLabel").textContent =
    `${file.name} · ${formatBytes(file.size)}`;

  document.getElementById("previewWrap").style.display = "block";
  document.getElementById("uploadZone").style.display  = "none";

  if (connected) habilitarBtn();
}

// ── Utilitários ───────────────────────────────────────────────────────────────
function formatBytes(bytes) {
  if (bytes < 1024)        return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function updateSlider(sliderId, labelId, value) {
  const label = document.getElementById(labelId);
  label.textContent = value;
  // Atualiza aria-valuenow para acessibilidade
  const slider = document.getElementById(sliderId);
  slider.setAttribute("aria-valuenow", slider.value);
}

function habilitarBtn() {
  const btn = document.getElementById("btnPredict");
  btn.disabled = false;
  document.getElementById("btnLabel").textContent = "Analisar imagem";
}

// ════════════════════════════════════════════════════
// PREDIÇÃO — envia imagem e renderiza resultado
// ════════════════════════════════════════════════════
async function predict() {
  if (!imgFile || !connected) return;

  // ── Estado de loading ───────────────────────────────────────────────────
  setLoading(true);

  const conf = document.getElementById("confSlider").value / 100;
  const iou  = document.getElementById("iouSlider").value  / 100;

  const form = new FormData();
  form.append("file", imgFile);

  try {
    const r = await fetch(`${API}/predict?conf=${conf}&iou=${iou}`, {
      method: "POST",
      body:   form,
    });

    if (!r.ok) {
      const msg = await r.text();
      throw new Error(`Erro ${r.status}: ${msg}`);
    }

    const data = await r.json();
    renderResultados(data);

  } catch (e) {
    // Mensagem amigável para o usuário
    const msg = e.message.includes("Failed to fetch")
      ? "Não foi possível conectar ao servidor. Verifique se o backend está rodando."
      : e.message;

    toast(msg);

  } finally {
    setLoading(false);
  }
}

function setLoading(loading) {
  const btn     = document.getElementById("btnPredict");
  const spinner = document.getElementById("spinner");
  const label   = document.getElementById("btnLabel");

  if (loading) {
    spinner.style.display = "block";
    label.textContent     = "Analisando...";
    btn.disabled          = true;
    btn.setAttribute("aria-busy", "true");
  } else {
    spinner.style.display = "none";
    label.textContent     = "Analisar novamente";
    btn.disabled          = false;
    btn.removeAttribute("aria-busy");
  }
}

// ════════════════════════════════════════════════════
// RENDERIZAÇÃO DOS RESULTADOS
// ════════════════════════════════════════════════════
function renderResultados(data) {
  // Atualiza imagem com bounding boxes
  const img = document.getElementById("previewImg");
  img.src = data.imagem_anotada;
  img.alt = `Resultado da análise: ${data.total_deteccoes} detecção(ões)`;

  // Badge de total
  const total = data.total_deteccoes;
  document.getElementById("badgeTotal").textContent =
    total + (total === 1 ? " detecção" : " detecções");

  // ── Lista de detecções ──────────────────────────────────────────────────
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
      const cls  = CLASSES[d.classe] || {
        cor: "var(--primary)", corBg: "var(--primary-light)",
        rotulo: d.classe, desc: "",
      };
      const pct  = Math.round(d.confianca * 100);
      const b    = d.bbox;
      const card = document.createElement("div");
      card.className = "det-card";
      card.setAttribute("role", "listitem");
      card.style.animationDelay = `${i * 60}ms`;
      card.style.borderLeftColor = cls.cor;
      card.style.borderLeftWidth = "3px";

      card.innerHTML = `
        <div class="class-dot"
             style="background:${cls.cor}"
             aria-hidden="true"></div>

        <div class="det-info">
          <div class="det-name" style="color:${cls.cor}">
            ${cls.rotulo}
            <span style="
              font-size:10px; font-weight:500;
              color:var(--muted); margin-left:6px;
            ">${cls.desc}</span>
          </div>
          <div class="det-meta">
            x1:${b.x1} y1:${b.y1} → x2:${b.x2} y2:${b.y2}
            &nbsp;·&nbsp; ${d.area_px.toLocaleString("pt-BR")} px²
          </div>
        </div>

        <div class="det-conf-col">
          <div class="conf-num" style="color:${cls.cor}">${pct}%</div>
          <div class="conf-lbl">confiança</div>
        </div>

        <div class="conf-bar-row" aria-hidden="true">
          <div class="conf-bar"
               style="width:${pct}%; background:${cls.cor}"></div>
        </div>
      `;

      list.appendChild(card);
    });
  }

  // ── Sumário por classe ──────────────────────────────────────────────────
  const count = { rim: 0, tumor: 0, cisto: 0 };
  data.deteccoes.forEach(d => {
    if (d.classe in count) count[d.classe]++;
  });

  document.getElementById("sumRim").textContent   = count.rim;
  document.getElementById("sumTumor").textContent = count.tumor;
  document.getElementById("sumCisto").textContent = count.cisto;

  const summaryPanel = document.getElementById("summaryPanel");
  summaryPanel.style.display = "block";
}

// ════════════════════════════════════════════════════
// TOAST — notificação de erro acessível
// ════════════════════════════════════════════════════
let toastTimer = null;

function toast(msg) {
  const t   = document.getElementById("toast");
  const msg_ = document.getElementById("toastMsg");

  msg_.textContent = msg;
  t.classList.add("show");

  // Cancela timer anterior se existir
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 5000);
}