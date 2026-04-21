const state = {
  events: [],
  lastMode: null,
};

const fmt = (value, digits = 2) => Number(value || 0).toFixed(digits);

function logEvent(message) {
  const stamp = new Date().toLocaleTimeString();
  state.events.unshift(`[${stamp}] ${message}`);
  state.events = state.events.slice(0, 8);
  document.querySelector("#event-log").textContent = state.events.join("\n");
}

async function setMode(mode) {
  const response = await fetch("/api/mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  if (!response.ok) {
    const error = await response.text();
    logEvent(`Mode request failed: ${error}`);
    return;
  }
  logEvent(`Requested ${mode.toUpperCase()} mode`);
  await refresh();
}

function render(data) {
  const metrics = data.metrics || {};
  const command = data.command || {};
  const vehicle = data.vehicle || {};
  const requested = data.requested_mode || "reactive";
  const active = data.active_mode || "reactive";

  document.querySelector("#active-mode").textContent = active;
  document.querySelector("#system-status").textContent = data.status || "unknown";
  document.querySelector("#elapsed").textContent = `${fmt(metrics.elapsed_s, 1)}s`;
  document.querySelector("#avg-speed").textContent = `${fmt(metrics.average_speed_mps)} m/s`;
  document.querySelector("#max-speed").textContent = `${fmt(metrics.max_speed_mps)} m/s`;
  document.querySelector("#collisions").textContent = metrics.collisions ?? 0;
  document.querySelector("#pos-x").textContent = fmt(vehicle.x);
  document.querySelector("#pos-y").textContent = fmt(vehicle.y);
  document.querySelector("#yaw").textContent = fmt(vehicle.yaw);
  document.querySelector("#vehicle-speed").textContent = fmt(vehicle.speed);
  document.querySelector("#cmd-speed").textContent = fmt(command.speed);
  document.querySelector("#cmd-steer").textContent = fmt(command.steering_angle, 3);

  const message = document.querySelector("#mode-message");
  if (data.last_error) {
    message.textContent = data.last_error;
  } else if (requested !== active) {
    message.textContent = `Switch requested: ${requested.toUpperCase()} waiting for ROS node`;
  } else {
    message.textContent = `${active.toUpperCase()} is active`;
  }

  document.querySelectorAll(".mode-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === active);
  });

  if (state.lastMode && state.lastMode !== active) {
    logEvent(`Active mode changed to ${active.toUpperCase()}`);
  }
  state.lastMode = active;
}

async function refresh() {
  const dot = document.querySelector("#connection-dot");
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    dot.classList.add("online");
    dot.classList.remove("offline");
    render(data);
  } catch (error) {
    dot.classList.remove("online");
    dot.classList.add("offline");
    document.querySelector("#system-status").textContent = "Dashboard offline";
  }
}

document.querySelectorAll(".mode-button").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

logEvent("Dashboard ready");
refresh();
setInterval(refresh, 750);

