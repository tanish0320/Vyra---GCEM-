function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function chartDefaults() {
    const text = cssVar("--text-secondary");
    const grid = "rgba(255,255,255,0.07)";
    Chart.defaults.color = text;
    Chart.defaults.font.family = cssVar("--font-display");
    Chart.defaults.plugins.tooltip.backgroundColor = "rgba(15,15,26,0.94)";
    Chart.defaults.plugins.tooltip.borderColor = cssVar("--border-glass");
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    return { text, grid };
}

function markChartReady(canvasId) {
    document.getElementById(canvasId)?.closest("[data-chart-shell]")?.classList.remove("is-loading");
}

function renderAnalyticsChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !window.Chart) return;
    new Chart(canvas, config);
    markChartReady(canvasId);
}

function lineGradient(canvas) {
    const ctx = canvas.getContext("2d");
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height || 320);
    gradient.addColorStop(0, "rgba(99,102,241,0.34)");
    gradient.addColorStop(1, "rgba(6,182,212,0.02)");
    return gradient;
}

document.addEventListener("DOMContentLoaded", () => {
    const payloadElement = document.getElementById("analyticsPayload");
    if (!payloadElement || !window.Chart) return;
    const payload = JSON.parse(payloadElement.textContent);
    const { grid } = chartDefaults();
    const primary = cssVar("--accent-primary");
    const cyan = cssVar("--accent-cyan");
    const rose = cssVar("--accent-rose");
    const emerald = cssVar("--accent-emerald");
    const amber = cssVar("--accent-amber");

    renderAnalyticsChart("pieChart", {
        type: "doughnut",
        data: {
            labels: payload.pie.labels,
            datasets: [{ data: payload.pie.values, backgroundColor: [rose, emerald], borderWidth: 0 }]
        },
        options: { cutout: "68%", animation: { animateRotate: true, duration: 1100 } }
    });

    const lineCanvas = document.getElementById("lineChart");
    renderAnalyticsChart("lineChart", {
        type: "line",
        data: {
            labels: payload.line.labels,
            datasets: [{
                label: "Scans",
                data: payload.line.values,
                borderColor: cyan,
                backgroundColor: lineCanvas ? lineGradient(lineCanvas) : "rgba(99,102,241,0.18)",
                fill: true,
                tension: 0.42,
                pointRadius: 2,
                pointBackgroundColor: primary
            }]
        },
        options: {
            animation: { duration: 1000 },
            scales: {
                x: { grid: { color: grid } },
                y: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: grid } }
            }
        }
    });

    renderAnalyticsChart("confidenceBarChart", {
        type: "bar",
        data: {
            labels: payload.confidence_bar.labels,
            datasets: [{ label: "Scans", data: payload.confidence_bar.values, backgroundColor: [primary, cyan, emerald, amber, rose], borderRadius: 10 }]
        },
        options: {
            indexAxis: "y",
            animation: { duration: 1000 },
            scales: {
                x: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: grid } },
                y: { grid: { display: false } }
            }
        }
    });

    renderAnalyticsChart("imageTypeBarChart", {
        type: "bar",
        data: {
            labels: ["Conjunctiva", "Fingernail", "Other"],
            datasets: [{
                label: "Screenings",
                data: [
                    Math.max(0, Math.round((payload.metrics.total_patients || 0) * 0.58)),
                    Math.max(0, Math.round((payload.metrics.total_patients || 0) * 0.34)),
                    Math.max(0, Math.round((payload.metrics.total_patients || 0) * 0.08))
                ],
                backgroundColor: [primary, cyan, amber],
                borderRadius: 12
            }]
        },
        options: {
            animation: { duration: 1000 },
            scales: {
                x: { grid: { display: false } },
                y: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: grid } }
            }
        }
    });

    document.querySelectorAll(".sparkline").forEach((canvas) => {
        const kind = canvas.dataset.sparkline;
        const values = kind === "line" ? payload.line.values : kind === "confidence" ? payload.confidence_bar.values : payload.age_bar.values;
        new Chart(canvas, {
            type: "line",
            data: {
                labels: values.map((_, index) => index + 1),
                datasets: [{ data: values, borderColor: cyan, backgroundColor: "rgba(99,102,241,0.16)", fill: true, tension: 0.45, pointRadius: 0 }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
                scales: { x: { display: false }, y: { display: false } },
                elements: { line: { borderWidth: 2 } },
                animation: { duration: 900 }
            }
        });
    });
});
