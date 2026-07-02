/* =========================================================
   AI Model Performance — dashboard rendering
   Reads the SAME JSON endpoints/fields as before:
     /static/retrieval_evaluation_results.json
     /static/llm_evaluation_results.json
   No backend / evaluation logic touched — presentation only.
   ========================================================= */

const ICONS = {
  disease_accuracy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="0.8" fill="currentColor"/></svg>',
  zone_accuracy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 21s-7-6.2-7-11.5A7 7 0 0 1 19 9.5C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.2"/></svg>',
  precision: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><circle cx="12" cy="12" r="6"/></svg>',
  recall: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-3-6.7"/><polyline points="21 3 21 9 15 9"/></svg>',
  f1_score: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="12" width="4" height="9" rx="1"/><rect x="10" y="7" width="4" height="14" rx="1"/><rect x="17" y="3" width="4" height="18" rx="1"/></svg>',
  average_response_time: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><polyline points="12 7 12 12 16 14"/></svg>',
};

const LABELS = {
  disease_accuracy: "Disease Accuracy",
  zone_accuracy: "Zone Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1_score: "F1 Score",
  average_response_time: "Response Time",
};

const RETRIEVAL_KEYS = ["disease_accuracy", "zone_accuracy", "precision", "recall", "f1_score","average_response_time"];
const LLM_KEYS = ["disease_accuracy", "zone_accuracy", "precision", "recall", "f1_score", "average_response_time"];

const RING_CIRCUMFERENCE = 150.8; // 2 * PI * r(24)

async function loadMetrics() {
  try {
    const [retrievalRes, llmRes] = await Promise.all([
      fetch("/static/retrieval_evaluation_results.json"),
      fetch("/static/llm_evaluation_results.json"),
    ]);
    const retrievalData = await retrievalRes.json();
    const llmData = await llmRes.json();

    buildCardSet("retrievalCards", retrievalData, RETRIEVAL_KEYS, "blue");
    buildCardSet("llmCards", llmData, LLM_KEYS, "teal");
    buildDataset(retrievalData, llmData);

    // animate rings/bars in after mount
    requestAnimationFrame(() => {
      setTimeout(animateAllCards, 60);
    });
  } catch (error) {
    console.error(error);
    document.querySelector(".container").insertAdjacentHTML(
      "beforeend",
      '<p class="load-error">Unable to load evaluation results.</p>'
    );
  }
}

function buildCardSet(containerId, data, keys, accent) {
  const el = document.getElementById(containerId);
  el.innerHTML = keys.map((key) => renderCard(key, data[key], accent)).join("");
}

function renderCard(key, rawValue, accent) {
  const isTime = key === "average_response_time";
  const value = Number(rawValue);
  const label = LABELS[key] || key;
  const icon = ICONS[key] || ICONS.disease_accuracy;
  const accentSolid = accent === "teal" ? "var(--teal)" : "var(--blue)";
  const accentGlow = accent === "teal" ? "rgba(22, 194, 166, 0.14)" : "rgba(45, 143, 224, 0.14)";

  if (isTime) {
    // scale against a practical max so both fast (retrieval) and
    // slower (LLM) response times read clearly on the same bar
    const maxScale = 15;
    const pct = Math.max(4, Math.min(100, (value / maxScale) * 100));
    return `
      <div class="metric-card" style="--card-accent:${accentGlow}; --card-accent-solid:${accentSolid}" data-bar="${pct}">
        <div class="metric-top">
          <div class="metric-icon">${icon}</div>
          <div class="metric-title">${label}</div>
        </div>
        <div class="metric-body">
          <div class="metric-value">${value.toFixed(2)}<span class="unit">sec</span></div>
        </div>
        <div class="metric-bar"><div class="metric-bar-fill"></div></div>
      </div>
    `;
  }

  const pct = Math.max(0, Math.min(100, value));
  return `
    <div class="metric-card" style="--card-accent:${accentGlow}; --card-accent-solid:${accentSolid}" data-ring="${pct}">
      <div class="metric-top">
        <div class="metric-icon">${icon}</div>
        <div class="metric-title">${label}</div>
      </div>
      <div class="metric-body">
        <div class="metric-value">${pct.toFixed(1)}<span class="unit">%</span></div>
        <div class="ring-wrap">
          <svg viewBox="0 0 58 58">
            <circle class="ring-track" cx="29" cy="29" r="24"></circle>
            <circle class="ring-fill" cx="29" cy="29" r="24"></circle>
          </svg>
        </div>
      </div>
    </div>
  `;
}

function animateAllCards() {
  document.querySelectorAll(".metric-card[data-ring]").forEach((card) => {
    const pct = parseFloat(card.dataset.ring);
    const ring = card.querySelector(".ring-fill");
    if (ring) {
      const offset = RING_CIRCUMFERENCE - (pct / 100) * RING_CIRCUMFERENCE;
      ring.style.strokeDashoffset = offset;
    }
  });
  document.querySelectorAll(".metric-card[data-bar]").forEach((card) => {
    const pct = parseFloat(card.dataset.bar);
    const bar = card.querySelector(".metric-bar-fill");
    if (bar) bar.style.width = pct + "%";
  });
}

function buildDataset(retrievalData, llmData) {
  document.getElementById("datasetCard").innerHTML = `
    <div class="dataset-box">
      <h3>Total Dataset</h3>
      <p>${Number(retrievalData.dataset_size).toLocaleString()}</p>
    </div>
    <div class="dataset-box">
      <h3>Retrieval Samples</h3>
      <p>${Number(retrievalData.samples_evaluated).toLocaleString()}</p>
    </div>
    <div class="dataset-box">
      <h3>LLM Samples</h3>
      <p>${Number(llmData.samples_evaluated).toLocaleString()}</p>
    </div>
    <div class="dataset-box">
      <h3>Diseases</h3>
      <p>15</p>
    </div>
    <div class="dataset-box">
      <h3>Language</h3>
      <p>Marathi</p>
    </div>
  `;
}

/* ---------- lightbox for confusion matrices ---------- */
function initLightbox() {
  const lightbox = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightboxImg");
  const closeBtn = document.getElementById("lightboxClose");

  document.querySelectorAll(".matrix-frame").forEach((frame) => {
    const img = frame.querySelector("img");
    if (!img) return;
    frame.addEventListener("click", () => {
      lightboxImg.src = img.dataset.full || img.src;
      lightboxImg.alt = img.alt;
      lightbox.classList.add("open");
    });
  });

  function close() {
    lightbox.classList.remove("open");
    lightboxImg.src = "";
  }

  closeBtn.addEventListener("click", close);
  lightbox.addEventListener("click", (e) => {
    if (e.target === lightbox) close();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") close();
  });
}

loadMetrics();
initLightbox();