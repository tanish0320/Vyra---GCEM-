/**
 * EDA & Research Findings Dashboard Interactions
 */
document.addEventListener("DOMContentLoaded", () => {
    // 1. Initialize Lucide Icons
    if (window.lucide) {
        window.lucide.createIcons();
    }

    // 2. Tab Navigation
    const tabBtns = document.querySelectorAll(".eda-tab-btn");
    const tabPanes = document.querySelectorAll(".eda-tab-pane");

    tabBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab-target");
            if (!targetId) return;

            tabBtns.forEach((b) => b.classList.remove("active"));
            tabPanes.forEach((p) => p.classList.remove("active"));

            btn.classList.add("active");
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.classList.add("active");
            }

            // Refresh icons in case dynamically shown
            if (window.lucide) {
                window.lucide.createIcons();
            }
        });
    });

    // 3. Lightbox Modal
    const lightbox = document.getElementById("edaLightbox");
    const lightboxImg = document.getElementById("lightboxImg");
    const lightboxTitle = document.getElementById("lightboxTitle");
    const lightboxCaption = document.getElementById("lightboxCaption");

    const openLightbox = (src, title, caption) => {
        if (!lightbox || !lightboxImg) return;
        lightboxImg.src = src;
        if (lightboxTitle) lightboxTitle.textContent = title || "Expanded Preview";
        if (lightboxCaption) lightboxCaption.textContent = caption || "";
        lightbox.classList.add("is-open");
        lightbox.setAttribute("aria-hidden", "false");
        document.body.style.overflow = "hidden";
        if (window.lucide) window.lucide.createIcons();
    };

    const closeLightbox = () => {
        if (!lightbox) return;
        lightbox.classList.remove("is-open");
        lightbox.setAttribute("aria-hidden", "true");
        document.body.style.overflow = "";
    };

    document.querySelectorAll("[data-zoom-img]").forEach((el) => {
        el.addEventListener("click", () => {
            const src = el.getAttribute("data-zoom-img");
            const title = el.getAttribute("data-zoom-title");
            const caption = el.getAttribute("data-zoom-caption");
            if (src) {
                openLightbox(src, title, caption);
            }
        });
    });

    document.querySelectorAll("[data-close-lightbox]").forEach((el) => {
        el.addEventListener("click", closeLightbox);
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && lightbox && lightbox.classList.contains("is-open")) {
            closeLightbox();
        }
    });

    // 4. Chart.js Initializations
    if (typeof Chart !== "undefined") {
        // Chart Defaults for dark theme
        Chart.defaults.color = "#94a3b8";
        Chart.defaults.font.family = "'Inter', system-ui, sans-serif";

        // Chart 1: Split Bar Chart (Modality Partitioning)
        const splitBarCtx = document.getElementById("splitBarChart");
        if (splitBarCtx) {
            new Chart(splitBarCtx, {
                type: "bar",
                data: {
                    labels: ["Train (3,560)", "Validation (602)", "Held-out Test (1,025)"],
                    datasets: [
                        {
                            label: "Palpebral Conjunctiva",
                            data: [653, 100, 174],
                            backgroundColor: "rgba(99, 102, 241, 0.8)",
                            borderColor: "#6366f1",
                            borderWidth: 1,
                            borderRadius: 6,
                        },
                        {
                            label: "Fingernail Bed",
                            data: [2907, 502, 851],
                            backgroundColor: "rgba(6, 182, 212, 0.8)",
                            borderColor: "#06b6d4",
                            borderWidth: 1,
                            borderRadius: 6,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: "top",
                            labels: {
                                boxWidth: 12,
                                usePointStyle: true,
                            },
                        },
                        tooltip: {
                            backgroundColor: "rgba(15, 15, 26, 0.95)",
                            borderColor: "rgba(255, 255, 255, 0.1)",
                            borderWidth: 1,
                            padding: 10,
                        },
                    },
                    scales: {
                        x: {
                            stacked: true,
                            grid: { display: false },
                        },
                        y: {
                            stacked: true,
                            grid: { color: "rgba(255, 255, 255, 0.05)" },
                        },
                    },
                },
            });
        }

        // Chart 2: Anemic vs Non-Anemic Donut
        const classDonutCtx = document.getElementById("classDonutChart");
        if (classDonutCtx) {
            new Chart(classDonutCtx, {
                type: "doughnut",
                data: {
                    labels: ["Anemic / High Risk (3,036)", "Non-Anemic / Healthy (2,151)"],
                    datasets: [
                        {
                            data: [3036, 2151],
                            backgroundColor: [
                                "rgba(244, 63, 94, 0.85)",
                                "rgba(16, 185, 129, 0.85)",
                            ],
                            borderColor: [
                                "#f43f5e",
                                "#10b981",
                            ],
                            borderWidth: 1,
                            hoverOffset: 6,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: "68%",
                    plugins: {
                        legend: {
                            position: "bottom",
                            labels: {
                                boxWidth: 12,
                                usePointStyle: true,
                                padding: 14,
                            },
                        },
                        tooltip: {
                            backgroundColor: "rgba(15, 15, 26, 0.95)",
                            borderColor: "rgba(255, 255, 255, 0.1)",
                            borderWidth: 1,
                            padding: 10,
                            callbacks: {
                                label: (ctx) => {
                                    const total = 5187;
                                    const val = ctx.raw || 0;
                                    const pct = ((val / total) * 100).toFixed(1);
                                    return ` ${ctx.label}: ${val} (${pct}%)`;
                                },
                            },
                        },
                    },
                },
            });
        }
    }
});
