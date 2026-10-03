// Demo dia cau 3D du doan do man DBSCL theo luoi H3.
// Khong dung Cesium Ion token: imagery = Esri World Imagery, terrain = mac dinh (Ellipsoid).

const API_BASE = "http://localhost:8002";

const CAMERA_VIEWS = {
  mekong: { lon: 105.5, lat: 9.6, height: 400000 },
  world: { lon: 20, lat: 10, height: 20000000 },
};

const MONTH_NAMES = [
  "Tháng 1", "Tháng 2", "Tháng 3", "Tháng 4", "Tháng 5", "Tháng 6",
  "Tháng 7", "Tháng 8", "Tháng 9", "Tháng 10", "Tháng 11", "Tháng 12",
];

// Color ramp giong voi css legend-gradient (dong bo tay).
const COLOR_STOPS = [
  { t: 0.0, color: Cesium.Color.fromCssColorString("#2166ac") },
  { t: 0.25, color: Cesium.Color.fromCssColorString("#67a9cf") },
  { t: 0.5, color: Cesium.Color.fromCssColorString("#fddbc7") },
  { t: 0.75, color: Cesium.Color.fromCssColorString("#d6604d") },
  { t: 1.0, color: Cesium.Color.fromCssColorString("#67001f") },
];

function salinityToColor(value, maxValue) {
  const t = maxValue > 0 ? Math.min(Math.max(value / maxValue, 0), 1) : 0;

  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    const a = COLOR_STOPS[i];
    const b = COLOR_STOPS[i + 1];
    if (t >= a.t && t <= b.t) {
      const localT = (t - a.t) / (b.t - a.t || 1);
      return Cesium.Color.lerp(a.color, b.color, localT, new Cesium.Color());
    }
  }
  return COLOR_STOPS[COLOR_STOPS.length - 1].color;
}

function salinityToCssColor(value, maxValue) {
  return salinityToColor(value, maxValue).toCssColorString();
}

// ---------------------------------------------------------------
// Cesium viewer
// ---------------------------------------------------------------

const viewer = new Cesium.Viewer("cesiumContainer", {
  // Anh ve tinh Esri World Imagery: khong can token, nhin thuc te hon OSM.
  baseLayer: new Cesium.ImageryLayer(
    new Cesium.UrlTemplateImageryProvider({
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      credit: "Esri, Maxar, Earthstar Geographics",
      maximumLevel: 18,
    })
  ),
  baseLayerPicker: false,
  geocoder: false,
  homeButton: false,
  sceneModePicker: false,
  navigationHelpButton: false,
  animation: false,
  timeline: false,
  fullscreenButton: false,
  selectionIndicator: false,
  infoBox: false,
});

// Lop nhan dia danh (ten tinh/huyen/song...) phu len anh ve tinh, cung tu Esri, khong can token.
viewer.imageryLayers.addImageryProvider(
  new Cesium.UrlTemplateImageryProvider({
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    credit: "Esri",
    maximumLevel: 18,
  })
);

function flyToScope(scope) {
  const view = CAMERA_VIEWS[scope];
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(view.lon, view.lat, view.height),
  });
}

flyToScope("mekong");

// ---------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------

const statusEl = document.getElementById("status");
const datePicker = document.getElementById("datePicker");
const predictBtn = document.getElementById("predictBtn");
const monthSlider = document.getElementById("monthSlider");
const monthLabel = document.getElementById("monthLabel");
const playBtn = document.getElementById("playBtn");
const stopBtn = document.getElementById("stopBtn");
const legendMaxEl = document.getElementById("legendMax");
const infoPopup = document.getElementById("infoPopup");
const infoPopupSwatch = document.getElementById("infoPopupSwatch");
const infoPopupContent = document.getElementById("infoPopupContent");
const infoPopupClose = document.getElementById("infoPopupClose");
const clearChartBtn = document.getElementById("clearChartBtn");
const viewHexStatsBtn = document.getElementById("viewHexStatsBtn");
const hexStatsPanel = document.getElementById("hexStatsPanel");
const hideHexStatsBtn = document.getElementById("hideHexStatsBtn");
const hexStatsHexId = document.getElementById("hexStatsHexId");
const hexStatsFrom = document.getElementById("hexStatsFrom");
const hexStatsTo = document.getElementById("hexStatsTo");
const hexStatsViewBtn = document.getElementById("hexStatsViewBtn");
const hexStatsClearBtn = document.getElementById("hexStatsClearBtn");
const hexStatsStatus = document.getElementById("hexStatsStatus");
const scopeSelect = document.getElementById("scopeSelect");
const scopeNote = document.getElementById("scopeNote");
const predictionModeBadge = document.getElementById("predictionModeBadge");
const predictionModeNote = document.getElementById("predictionModeNote");

function updatePredictionModeBadge(modelMeta) {
  if (!modelMeta) return;
  const isModel = modelMeta.prediction_mode === "model";
  predictionModeBadge.textContent = isModel
    ? `✓ MODEL — ${modelMeta.model_name || "?"} (${modelMeta.model_version || "?"})`
    : "⚠ MOCK — dữ liệu mô phỏng";
  predictionModeBadge.classList.toggle("mode-badge-model", isModel);
  predictionModeBadge.classList.toggle("mode-badge-mock", !isModel);
  predictionModeNote.textContent = isModel
    ? "Kết quả từ model đã train — vẫn cần đối chiếu độ chính xác (metrics) trước khi trích dẫn khoa học."
    : "Chưa phải mô hình dự đoán thật, không dùng để kết luận khoa học.";
}

let currentMaxSalinity = 0;
let playTimer = null;
let selectedHexId = null;
let currentScope = "mekong";

function todayISO() {
  const now = new Date();
  return now.toISOString().slice(0, 10);
}

const today = new Date();
datePicker.value = todayISO();
monthSlider.value = today.getMonth() + 1;
monthLabel.textContent = `${MONTH_NAMES[today.getMonth()]}/${today.getFullYear()}`;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

// ---------------------------------------------------------------
// Ve luoi hex len globe
// ---------------------------------------------------------------

function clearHexEntities() {
  viewer.entities.values
    .filter((e) => e.isHexCell)
    .forEach((e) => viewer.entities.remove(e));
}

function computeAverage(features) {
  if (!features.length) return 0;
  const sum = features.reduce((acc, f) => acc + f.properties.salinity_ppt, 0);
  return sum / features.length;
}

function renderFeatureCollection(featureCollection) {
  clearHexEntities();

  const values = featureCollection.features.map((f) => f.properties.salinity_ppt);
  currentMaxSalinity = values.length ? Math.max(...values) : 0;
  legendMaxEl.textContent = currentMaxSalinity.toFixed(1);

  for (const feature of featureCollection.features) {
    const ring = feature.geometry.coordinates[0]; // [[lon, lat], ...]
    const flat = [];
    for (const [lon, lat] of ring) {
      flat.push(lon, lat);
    }

    const salinity = feature.properties.salinity_ppt;
    const color = salinityToColor(salinity, currentMaxSalinity);

    const entity = viewer.entities.add({
      polygon: {
        hierarchy: Cesium.Cartesian3.fromDegreesArray(flat),
        material: color.withAlpha(0.65),
        outline: true,
        outlineColor: Cesium.Color.WHITE.withAlpha(0.3),
        height: 0,
      },
      properties: {
        h3_index: feature.properties.h3_index,
        salinity_ppt: salinity,
        date: feature.properties.date,
      },
    });
    entity.isHexCell = true;
  }
}

// ---------------------------------------------------------------
// Goi API + dieu phoi chung (date picker, slider, play)
// ---------------------------------------------------------------

async function fetchPrediction(date) {
  const url = `${API_BASE}/api/predict?date=${encodeURIComponent(date)}&scope=${currentScope}`;
  const response = await fetch(url);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Loi API: HTTP ${response.status}`);
  }
  return response.json();
}

async function loadDate(date, { silentButtons = false } = {}) {
  if (!silentButtons) predictBtn.disabled = true;
  setStatus(`Đang tải dữ liệu dự đoán cho ${date}...`);

  try {
    const featureCollection = await fetchPrediction(date);
    renderFeatureCollection(featureCollection);
    updatePredictionModeBadge(featureCollection.model_meta);
    const avg = computeAverage(featureCollection.features);
    addChartPoint(date, avg);
    setStatus(
      `Đã hiển thị ${featureCollection.features.length} ô lục giác cho ngày ${date} (TB: ${avg.toFixed(2)} ppt).`
    );
    return true;
  } catch (err) {
    setStatus(err.message || "Không thể tải dữ liệu từ API.", true);
    return false;
  } finally {
    if (!silentButtons) predictBtn.disabled = false;
  }
}

scopeSelect.addEventListener("change", () => {
  stopPlaying();
  currentScope = scopeSelect.value;
  scopeNote.classList.toggle("hidden", currentScope !== "world");

  // Doi pham vi thi cac diem bieu do/hex dang chon o pham vi cu khong con
  // y nghia nua -> reset sach de tranh nham lan du lieu giua 2 scope.
  selectedHexId = null;
  infoPopup.classList.add("hidden");
  hexStatsPanel.classList.add("hidden");
  chartPoints.clear();
  salinityChart.data.labels = [];
  salinityChart.data.datasets[0].data = [];
  salinityChart.update();

  flyToScope(currentScope);
  loadDate(datePicker.value);
});

predictBtn.addEventListener("click", () => {
  const date = datePicker.value;
  if (!date) {
    setStatus("Vui lòng chọn ngày.", true);
    return;
  }
  stopPlaying();
  const month = Number(date.split("-")[1]);
  monthSlider.value = month;
  monthLabel.textContent = `${MONTH_NAMES[month - 1]}/${date.split("-")[0]}`;
  loadDate(date);
});

// ---------------------------------------------------------------
// Thanh truot theo thang + tua nhanh
// ---------------------------------------------------------------

function currentSliderYear() {
  const yearFromPicker = Number(datePicker.value?.split("-")[0]);
  return yearFromPicker || today.getFullYear();
}

function dateFromMonth(month) {
  const year = currentSliderYear();
  return `${year}-${String(month).padStart(2, "0")}-15`;
}

monthSlider.addEventListener("input", () => {
  const month = Number(monthSlider.value);
  monthLabel.textContent = `${MONTH_NAMES[month - 1]}/${currentSliderYear()}`;
});

monthSlider.addEventListener("change", () => {
  const month = Number(monthSlider.value);
  loadDate(dateFromMonth(month), { silentButtons: true });
});

function stopPlaying() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
  playBtn.classList.remove("hidden");
  stopBtn.classList.add("hidden");
}

function startPlaying() {
  playBtn.classList.add("hidden");
  stopBtn.classList.remove("hidden");

  playTimer = setInterval(async () => {
    let month = Number(monthSlider.value) + 1;
    if (month > 12) month = 1;
    monthSlider.value = month;
    monthLabel.textContent = `${MONTH_NAMES[month - 1]}/${currentSliderYear()}`;
    await loadDate(dateFromMonth(month), { silentButtons: true });
  }, 1200);
}

playBtn.addEventListener("click", startPlaying);
stopBtn.addEventListener("click", stopPlaying);

// ---------------------------------------------------------------
// Bieu do xu huong (Chart.js)
// ---------------------------------------------------------------

const chartPoints = new Map(); // date -> avg salinity

const salinityChartCtx = document.getElementById("salinityChart").getContext("2d");
const salinityChart = new Chart(salinityChartCtx, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "Độ mặn TB (ppt)",
        data: [],
        borderColor: "#67a9cf",
        backgroundColor: "rgba(103, 169, 207, 0.25)",
        tension: 0.3,
        fill: true,
        pointRadius: 3,
        pointBackgroundColor: "#fddbc7",
      },
    ],
  },
  options: {
    responsive: true,
    plugins: {
      legend: { labels: { color: "#f2f2f2", font: { size: 10 } } },
    },
    scales: {
      x: { ticks: { color: "#a9b2c3", font: { size: 9 } }, grid: { color: "rgba(255,255,255,0.06)" } },
      y: { ticks: { color: "#a9b2c3", font: { size: 9 } }, grid: { color: "rgba(255,255,255,0.06)" }, beginAtZero: true },
    },
  },
});

function addChartPoint(date, avgValue) {
  chartPoints.set(date, avgValue);
  const sortedDates = Array.from(chartPoints.keys()).sort();
  salinityChart.data.labels = sortedDates;
  salinityChart.data.datasets[0].data = sortedDates.map((d) => Number(chartPoints.get(d).toFixed(2)));
  salinityChart.update();
}

clearChartBtn.addEventListener("click", () => {
  chartPoints.clear();
  salinityChart.data.labels = [];
  salinityChart.data.datasets[0].data = [];
  salinityChart.update();
});

// ---------------------------------------------------------------
// Popup thong tin khi click 1 o hex
// ---------------------------------------------------------------

const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
handler.setInputAction((movement) => {
  const picked = viewer.scene.pick(movement.position);
  if (Cesium.defined(picked) && picked.id && picked.id.isHexCell) {
    const props = picked.id.properties;
    const salinity = props.salinity_ppt.getValue();
    const color = salinityToCssColor(salinity, currentMaxSalinity);
    const pct = currentMaxSalinity > 0 ? Math.min((salinity / currentMaxSalinity) * 100, 100) : 0;

    selectedHexId = props.h3_index.getValue();

    infoPopupSwatch.style.background = color;
    infoPopupContent.innerHTML = `
      <div class="info-row"><span class="k">Mã ô H3</span><span class="v">${selectedHexId}</span></div>
      <div class="info-row"><span class="k">Độ mặn</span><span class="v">${salinity} ppt</span></div>
      <div class="info-row"><span class="k">Ngày</span><span class="v">${props.date.getValue()}</span></div>
      <div class="salinity-bar-track">
        <div class="salinity-bar-fill" style="width:${pct}%; background:${color};"></div>
      </div>
    `;
    infoPopup.classList.remove("hidden");
  }
}, Cesium.ScreenSpaceEventType.LEFT_CLICK);

infoPopupClose.addEventListener("click", () => {
  infoPopup.classList.add("hidden");
});

// ---------------------------------------------------------------
// Panel thong ke xu huong theo 1 dia diem (hex) cu the
// ---------------------------------------------------------------

const hexTrendChartCtx = document.getElementById("hexTrendChart").getContext("2d");
const hexTrendChart = new Chart(hexTrendChartCtx, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "Độ mặn (ppt)",
        data: [],
        borderColor: "#d6604d",
        backgroundColor: "rgba(214, 96, 77, 0.25)",
        tension: 0.3,
        fill: true,
        pointRadius: 3,
        pointBackgroundColor: "#fddbc7",
      },
    ],
  },
  options: {
    responsive: true,
    plugins: {
      legend: { labels: { color: "#f2f2f2", font: { size: 10 } } },
    },
    scales: {
      x: { ticks: { color: "#a9b2c3", font: { size: 9 } }, grid: { color: "rgba(255,255,255,0.06)" } },
      y: { ticks: { color: "#a9b2c3", font: { size: 9 } }, grid: { color: "rgba(255,255,255,0.06)" }, beginAtZero: true },
    },
  },
});

function setHexStatsStatus(message, isError = false) {
  hexStatsStatus.textContent = message;
  hexStatsStatus.classList.toggle("error", isError);
}

function defaultHexStatsRange() {
  const to = todayISO();
  const oneYearAgo = new Date();
  oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);
  const from = oneYearAgo.toISOString().slice(0, 10);
  return { from, to };
}

function resetHexTrendChart() {
  hexTrendChart.data.labels = [];
  hexTrendChart.data.datasets[0].data = [];
  hexTrendChart.update();
}

viewHexStatsBtn.addEventListener("click", () => {
  if (!selectedHexId) return;
  hexStatsHexId.textContent = selectedHexId;
  const { from, to } = defaultHexStatsRange();
  hexStatsFrom.value = from;
  hexStatsTo.value = to;
  setHexStatsStatus("");
  resetHexTrendChart();
  hexStatsPanel.classList.remove("hidden");
});

hideHexStatsBtn.addEventListener("click", () => {
  hexStatsPanel.classList.add("hidden");
});

hexStatsViewBtn.addEventListener("click", async () => {
  if (!selectedHexId) return;
  const from = hexStatsFrom.value;
  const to = hexStatsTo.value;
  if (!from || !to) {
    setHexStatsStatus("Vui lòng chọn đủ ngày bắt đầu và kết thúc.", true);
    return;
  }
  if (from > to) {
    setHexStatsStatus("Ngày bắt đầu phải trước ngày kết thúc.", true);
    return;
  }

  hexStatsViewBtn.disabled = true;
  setHexStatsStatus("Đang tải dữ liệu...");

  try {
    const url = `${API_BASE}/api/predict/series?h3_index=${encodeURIComponent(selectedHexId)}&from=${from}&to=${to}&scope=${currentScope}`;
    const response = await fetch(url);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Lỗi API: HTTP ${response.status}`);
    }
    const data = await response.json();

    hexTrendChart.data.labels = data.points.map((p) => p.date);
    hexTrendChart.data.datasets[0].data = data.points.map((p) => p.salinity_ppt);
    hexTrendChart.update();

    setHexStatsStatus(`Đã hiển thị ${data.points.length} điểm dữ liệu.`);
  } catch (err) {
    setHexStatsStatus(err.message || "Không thể tải dữ liệu.", true);
  } finally {
    hexStatsViewBtn.disabled = false;
  }
});

hexStatsClearBtn.addEventListener("click", () => {
  const { from, to } = defaultHexStatsRange();
  hexStatsFrom.value = from;
  hexStatsTo.value = to;
  setHexStatsStatus("");
  resetHexTrendChart();
});

// Tai du lieu lan dau ngay khi trang mo.
loadDate(datePicker.value);
