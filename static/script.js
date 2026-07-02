/* =========================================================
   AI Symptom Checker — script.js
   Manages: complaint -> analyze -> follow-up loop -> completion
   Talks to: POST /api/analyze  { complaint, conversation_history }
   Never refreshes the page. Always replaces the current result card.
   ========================================================= */

(() => {
  const els = {
    emergencyBanner: document.getElementById("emergencyBanner"),
    complaintCard: document.getElementById("complaintCard"),
    complaintInput: document.getElementById("complaintInput"),
    analyzeBtn: document.getElementById("analyzeBtn"),
    resetBtn: document.getElementById("resetBtn"),
    errorText: document.getElementById("errorText"),

    resultCard: document.getElementById("resultCard"),
    zoneBadge: document.getElementById("zoneBadge"),
    diseaseTag: document.getElementById("diseaseTag"),
    symptomsLine: document.getElementById("symptomsLine"),
    actionLine: document.getElementById("actionLine"),

    followupCard: document.getElementById("followupCard"),
    progressLabel: document.getElementById("progressLabel"),
    progressFill: document.getElementById("progressFill"),
    followupQuestionText: document.getElementById("followupQuestionText"),
    followupAnswer: document.getElementById("followupAnswer"),
    submitAnswerBtn: document.getElementById("submitAnswerBtn"),

  };

  // ---- conversation state (kept internally, not necessarily shown) ----
  let state = {
    complaint: "",
    history: [],          // [{question, answer}, ...]
    lastQuestion: "",      // question currently being answered
    busy: false,
  };

  function resetAll() {
    state = { complaint: "", history: [], lastQuestion: "", busy: false };
    els.complaintInput.value = "";
    hide(els.resultCard);
    hide(els.followupCard);
    hideError();
    setButtonLoading(els.analyzeBtn, false);
    setButtonLoading(els.submitAnswerBtn, false);
    els.complaintInput.disabled = false;
    els.complaintInput.focus();
  }

  function hide(el) { el.hidden = true; }
  function show(el) { el.hidden = false; }

  function hideError() {
    els.errorText.hidden = true;
    els.errorText.textContent = "";
  }
  function showError(msg) {
    els.errorText.textContent = msg;
    els.errorText.hidden = false;
  }

  function setButtonLoading(btn, loading) {
    const spinner = btn.querySelector(".spinner");
    const label = btn.querySelector(".btn-label");
    btn.disabled = loading;
    if (spinner) spinner.hidden = !loading;
    if (label) label.style.opacity = loading ? "0.55" : "1";
  }

  function zoneClass(zone) {
    const map = { Red: "zone-red", Orange: "zone-orange", Yellow: "zone-yellow", Green: "zone-green" };
    return map[zone] || "zone-yellow";
  }

  function zoneEmoji(zone) {
    const map = { Red: "🔴", Orange: "🟠", Yellow: "🟡", Green: "🟢" };
    return map[zone] || "🟡";
  }

  async function callAnalyze() {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        complaint: state.complaint,
        conversation_history: state.history,
      }),
    });

    if (!res.ok) {
      let detail = "काहीतरी चूक झाली. कृपया पुन्हा प्रयत्न करा.";
      try {
        const data = await res.json();
        if (data && data.detail) detail = data.detail;
      } catch (_) {}
      throw new Error(detail);
    }

    return res.json();
  }

  function renderResult(data) {
    console.log("renderResult() called");
    console.log(data);
    console.log(els.emergencyBanner);

    show(els.resultCard);
    els.zoneBadge.className = "zone-badge " + zoneClass(data.zone);
    els.zoneBadge.textContent = `${zoneEmoji(data.zone)} ${data.zone} · ${data.zone_label}`;

    els.symptomsLine.textContent = data.patient_symptoms_line || "";
    els.actionLine.textContent = data.patient_action_line || "";
    document.getElementById("confidenceValue").textContent =
    `${data.confidence.toFixed(1)}%`;

    if (data.zone === "Red") {
        console.log("RED DETECTED");
        els.emergencyBanner.hidden = false;
    } else {
        console.log("NOT RED");
        els.emergencyBanner.hidden = true;
    }

  }

  function renderFollowup(data) {
    const used = data.followups_used;
    const max = data.max_followups;
    const question = (data.followup_question || "").trim();

    if (!question) {
        hide(els.followupCard);
        els.followupAnswer.disabled = true;
        return;
    }

    show(els.followupCard);

    state.lastQuestion = question;
    els.followupQuestionText.textContent = question;
    els.followupAnswer.value = "";
    els.followupAnswer.disabled = false;

    const currentQNum = Math.min(used + 1, max);
    els.progressLabel.textContent = `Question ${currentQNum} / ${max}`;
    const pct = Math.min(100, Math.round((currentQNum / max) * 100));
    els.progressFill.style.width = pct + "%";

    els.followupAnswer.focus();
  }

  async function handleAnalyzeClick() {
    if (state.busy) return;
    const text = els.complaintInput.value.trim();
    if (!text) {
      showError("कृपया तुमची तक्रार लिहा.");
      return;
    }

    hideError();
    state.complaint = text;
    state.history = [];
    state.busy = true;

    hide(els.resultCard);
    hide(els.followupCard);
    setButtonLoading(els.analyzeBtn, true);
    els.complaintInput.disabled = true;

    try {
        const data = await callAnalyze();

        if ((data.followup_question || "").trim()) {
            hide(els.resultCard);
            renderFollowup(data);
        } else {
            renderResult(data);
            hide(els.followupCard);
        }
    } catch (err) {
      showError(err.message || "विश्लेषण अयशस्वी झाले.");
    } finally {
      state.busy = false;
      setButtonLoading(els.analyzeBtn, false);
    }
  }

  async function handleSubmitAnswer() {
    if (state.busy) return;
    const answer = els.followupAnswer.value.trim();
    if (!answer) {
      els.followupAnswer.focus();
      return;
    }

    state.busy = true;
    setButtonLoading(els.submitAnswerBtn, true);
    els.followupAnswer.disabled = true;

    // append this Q/A to the conversation history (Option A)
    state.history.push({ question: state.lastQuestion, answer });

    try {
        const data = await callAnalyze();

        if ((data.followup_question || "").trim()) {
            hide(els.resultCard);
            renderFollowup(data);
        } else {
            renderResult(data);
            hide(els.followupCard);
        }
    } catch (err) {
      // roll back the optimistic push so the user can retry
      state.history.pop();
      showError(err.message || "उत्तर पाठवण्यात अडचण आली.");
      els.followupAnswer.disabled = false;
    } finally {
      state.busy = false;
      setButtonLoading(els.submitAnswerBtn, false);
    }
  }

  els.analyzeBtn.addEventListener("click", handleAnalyzeClick);

  els.resetBtn.addEventListener("click", resetAll);
  els.submitAnswerBtn.addEventListener("click", handleSubmitAnswer);

  els.complaintInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAnalyzeClick();
    }
  });
  els.followupAnswer.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSubmitAnswer();
    }
  });
})();