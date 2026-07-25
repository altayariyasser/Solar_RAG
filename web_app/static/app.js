const form = document.querySelector("#questionForm");
const input = document.querySelector("#questionInput");
const submitButton = form.querySelector("button[type='submit']");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcomeMessage");
const snapshot = document.querySelector("#snapshot");
const drivers = document.querySelector("#drivers");
const weatherDrivers = document.querySelector("#weatherDrivers");
const airDrivers = document.querySelector("#airDrivers");
const sourceBadge = document.querySelector("#sourceBadge");
const toast = document.querySelector("#toast");

let context = {};
const API_BASE =
  window.location.hostname === "altayariyasser.github.io"
    ? "https://solar-rag.onrender.com"
    : "";

const weatherFields = {
  temperature_2m_mean: ["Mean temperature", "°C", 1],
  relative_humidity_2m_mean: ["Relative humidity", "%", 0],
  wind_speed_10m_mean: ["Wind speed", "km/h", 1],
  cloud_cover_mean: ["Cloud cover", "%", 0],
  precipitation_sum: ["Precipitation", "mm", 1],
  shortwave_radiation_sum: ["Solar radiation", "MJ/m²", 1],
  sunshine_duration: ["Sunshine", "hours", 1],
};

const airFields = {
  pm10: ["PM10", "µg/m³", 1],
  pm2_5: ["PM2.5", "µg/m³", 1],
  carbon_monoxide: ["Carbon monoxide", "µg/m³", 1],
  nitrogen_dioxide: ["Nitrogen dioxide", "µg/m³", 1],
  ozone: ["Ozone", "µg/m³", 1],
  sulphur_dioxide: ["Sulphur dioxide", "µg/m³", 1],
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function friendlyDate(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("en", {
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${value}T00:00:00Z`));
}

function addMessage(role, text, extraClass = "") {
  const row = document.createElement("div");
  row.className = `message ${role} ${extraClass}`.trim();
  row.innerHTML = `<div class="message-bubble">${escapeHtml(text)}</div>`;
  messages.appendChild(row);
  row.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return row;
}

function addTyping() {
  const row = document.createElement("div");
  row.className = "message assistant";
  row.innerHTML =
    '<div class="message-bubble typing"><i></i><i></i><i></i></div>';
  messages.appendChild(row);
  return row;
}

function riskClass(label) {
  const normalized = String(label).toLowerCase();
  if (normalized === "good") return "risk-good";
  if (normalized === "moderate") return "risk-moderate";
  return "risk-high";
}

function renderSnapshot(result) {
  const prediction = result.predictions || {};
  const solar = Number(prediction.solar_output_kwh || 0);
  const aqi = Number(prediction.aqi_value || 0);
  const risk = prediction.aqi_risk_level || "Unavailable";
  const position = result.business?.label || "Model estimate";
  const positionNote =
    result.business?.detail || "Portfolio benchmark unavailable";

  snapshot.innerHTML = `
    <div class="result-header">
      <span>${escapeHtml(result.source_kind || "model")} analysis</span>
      <h3>${escapeHtml(result.city)} · ${escapeHtml(friendlyDate(result.date))}</h3>
    </div>
    <div class="metric-stack">
      <article class="result-metric">
        <span>Estimated solar energy</span>
        <strong>${solar.toFixed(1)} kWh</strong>
        <small>Daily modeled output</small>
      </article>
      <article class="result-metric">
        <span>Portfolio position</span>
        <strong>${escapeHtml(position)}</strong>
        <small>${escapeHtml(positionNote)}</small>
      </article>
      <article class="result-metric">
        <span>Air quality</span>
        <strong class="${riskClass(risk)}">${escapeHtml(risk)}</strong>
        <small>AQI ${aqi.toFixed(0)}</small>
      </article>
    </div>
  `;
}

function formatDriver(key, value, specification) {
  let number = Number(value);
  const [label, unit, decimals] = specification;
  if (key === "sunshine_duration") number /= 3600;
  return `
    <article class="driver-item">
      <span>${escapeHtml(label)}</span>
      <strong>${Number.isFinite(number) ? number.toFixed(decimals) : "—"} ${unit}</strong>
    </article>
  `;
}

function renderDriverGroup(element, data, fields) {
  const content = Object.entries(fields)
    .filter(([key]) => data[key] !== undefined && data[key] !== null)
    .map(([key, specification]) =>
      formatDriver(key, data[key], specification),
    )
    .join("");
  element.innerHTML =
    content || '<p class="composer-help">No measurements available.</p>';
}

function renderDrivers(result) {
  renderDriverGroup(weatherDrivers, result.data || {}, weatherFields);
  renderDriverGroup(airDrivers, result.data || {}, airFields);
  sourceBadge.textContent = result.source_label || "Model inputs";
  drivers.classList.remove("hidden");
}

function showToast(text) {
  toast.textContent = text;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 2600);
}

async function runAnalysis(question) {
  welcome.hidden = true;
  addMessage("user", question);
  const typing = addTyping();
  submitButton.disabled = true;
  input.disabled = true;

  try {
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, context }),
    });
    const result = await response.json();
    typing.remove();

    if (!response.ok || result.status !== "success") {
      addMessage(
        "assistant",
        result.error || "I could not complete that analysis.",
        "error",
      );
      return;
    }

    addMessage("assistant", result.summary || "Analysis completed.");
    context = {
      city: result.city,
      date: result.date,
      intents: result.intents,
    };
    renderSnapshot(result);
    renderDrivers(result);
  } catch (_error) {
    typing.remove();
    addMessage(
      "assistant",
      "The model service is starting or temporarily unavailable. On the free service, the first request can take about one minute—please try again shortly.",
      "error",
    );
  } finally {
    submitButton.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  input.style.height = "auto";
  runAnalysis(question);
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.question;
    input.focus();
  });
});

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => {
    document
      .querySelectorAll(".nav-item")
      .forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    document.querySelector(`#${button.dataset.target}`)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
    document.body.classList.remove("menu-open");
  });
});

function clearAnalysis() {
  messages.innerHTML = "";
  welcome.hidden = false;
  context = {};
  drivers.classList.add("hidden");
  snapshot.innerHTML = `
    <div class="empty-state">
      <span class="empty-icon" aria-hidden="true">☀</span>
      <h3>Analysis snapshot</h3>
      <p>Your key results will appear here after you ask a question.</p>
    </div>
  `;
  input.value = "";
  input.focus();
  showToast("Analysis cleared");
}

document.querySelector(".clear-button").addEventListener("click", clearAnalysis);
document.querySelector(".new-analysis").addEventListener("click", () => {
  clearAnalysis();
  document.querySelector("#analysis").scrollIntoView({ behavior: "smooth" });
});
document.querySelector(".menu-button").addEventListener("click", () => {
  document.body.classList.toggle("menu-open");
});
