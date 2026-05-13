const API = window.location.origin;

let imgFile = null;
let connected = false;

const COLORS = {
  rim: "var(--rim)",
  tumor: "var(--tumor)",
  cisto: "var(--cisto)"
};

async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);

    if (!r.ok) throw new Error();

    connected = true;

    document.getElementById("statusPill").classList.add("online");
    document.getElementById("statusText").textContent = "backend online";

    if (imgFile) habilitarBtn();

  } catch {
    document.getElementById("statusText").textContent = "sem conexão";

    setTimeout(checkHealth, 3000);
  }
}

window.addEventListener("load", checkHealth);

/* DRAG & DROP */

const zone = document.getElementById("uploadZone");

zone.addEventListener("dragover", e => {
  e.preventDefault();
  zone.classList.add("drag");
});

zone.addEventListener("dragleave", () => {
  zone.classList.remove("drag");
});

zone.addEventListener("drop", e => {
  e.preventDefault();

  zone.classList.remove("drag");

  const f = e.dataTransfer.files[0];

  if (f?.type.startsWith("image/")) {
    processFile(f);
  }
});

function onFile(input) {
  if (input.files[0]) {
    processFile(input.files[0]);
  }
}

function processFile(file) {
  imgFile = file;

  document.getElementById("previewImg").src =
    URL.createObjectURL(file);

  document.getElementById("previewWrap").style.display = "block";

  document.getElementById("uploadZone").style.display = "none";

  if (connected) habilitarBtn();
}

function habilitarBtn() {
  document.getElementById("btnPredict").disabled = false;

  document.getElementById("btnLabel").textContent =
    "Analisar imagem";
}

/* PREDIÇÃO */

async function predict() {

  if (!imgFile || !connected) return;

  document.getElementById("spinner").style.display = "block";

  document.getElementById("btnLabel").textContent =
    "Analisando...";

  document.getElementById("btnPredict").disabled = true;

  const conf =
    document.getElementById("confSlider").value / 100;

  const iou =
    document.getElementById("iouSlider").value / 100;

  const form = new FormData();

  form.append("file", imgFile);

  try {

    const r = await fetch(
      `${API}/predict?conf=${conf}&iou=${iou}`,
      {
        method: "POST",
        body: form
      }
    );

    if (!r.ok) {
      throw new Error(await r.text());
    }

    const data = await r.json();

    renderResultados(data);

  } catch (e) {

    toast(e.message);

  } finally {

    document.getElementById("spinner").style.display = "none";

    document.getElementById("btnLabel").textContent =
      "Analisar novamente";

    document.getElementById("btnPredict").disabled = false;
  }
}

/* RESULTADOS */

function renderResultados(data) {

  document.getElementById("previewImg").src =
    data.imagem_anotada;

  document.getElementById("badgeTotal").textContent =
    `${data.total_deteccoes} detecções`;

  const list = document.getElementById("resultsList");

  list.innerHTML = "";

  data.deteccoes.forEach(d => {

    const cor = COLORS[d.classe] || "var(--accent)";

    const pct = Math.round(d.confianca * 100);

    const card = document.createElement("div");

    card.className = "det-card";

    card.innerHTML = `
      <div class="det-name" style="color:${cor}">
        ${d.classe}
      </div>

      <div>
        Confiança: ${pct}%
      </div>
    `;

    list.appendChild(card);
  });
}

/* TOAST */

function toast(msg) {

  const t = document.getElementById("toast");

  t.textContent = msg;

  t.classList.add("show");

  setTimeout(() => {
    t.classList.remove("show");
  }, 4000);
}