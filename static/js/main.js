function updateCurrentDateTime() {
    const target = document.getElementById("currentDateTime");
    if (!target) return;

    target.textContent = new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short"
    }).format(new Date());
}

function throttle(callback, wait = 100) {
    let waiting = false;
    return (...args) => {
        if (waiting) return;
        waiting = true;
        requestAnimationFrame(() => {
            callback(...args);
            window.setTimeout(() => {
                waiting = false;
            }, wait);
        });
    };
}

function debounce(callback, wait = 300) {
    let timer = 0;
    return (...args) => {
        window.clearTimeout(timer);
        timer = window.setTimeout(() => callback(...args), wait);
    };
}

function bindNavbar() {
    const header = document.querySelector(".site-header");
    const openButton = document.querySelector("[data-drawer-open]");
    const closeButton = document.querySelector("[data-drawer-close]");
    const drawer = document.getElementById("mobileDrawer");
    const backdrop = document.getElementById("drawerBackdrop");

    const updateScrolled = () => {
        header?.classList.toggle("is-scrolled", window.scrollY > 50);
    };

    const close = () => {
        drawer?.classList.remove("open");
        backdrop?.classList.remove("show");
        document.body.classList.remove("nav-open");
    };

    const open = () => {
        drawer?.classList.add("open");
        backdrop?.classList.add("show");
        document.body.classList.add("nav-open");
    };

    window.addEventListener("scroll", throttle(updateScrolled, 80), { passive: true });
    updateScrolled();
    openButton?.addEventListener("click", open);
    closeButton?.addEventListener("click", close);
    backdrop?.addEventListener("click", close);
    drawer?.querySelectorAll("a").forEach((link) => link.addEventListener("click", close));
}

function bindLiveStatus() {
    const status = document.getElementById("liveStatus");
    if (!status) return;

    const label = status.querySelector("[data-status-label]");

    const setState = (state, text) => {
        status.classList.toggle("is-warning", state === "warning");
        status.classList.toggle("is-offline", state === "offline");
        if (label) label.textContent = text;
    };

    const poll = async () => {
        try {
            const response = await fetch("/health", { headers: { Accept: "application/json" }, cache: "no-store" });
            if (!response.ok) {
                setState("offline", "System Issue");
                return;
            }
            const payload = await response.json();
            if (payload.lightweight_ui) {
                setState("online", "UI Online");
            } else if (payload.predictor_loaded) {
                setState("online", "System Online");
            } else if (payload.checkpoint_available) {
                setState("warning", "Model Waking");
            } else {
                setState("offline", "Weights Missing");
            }
        } catch (error) {
            setState("offline", "Offline");
        }
    };

    poll();
    window.setInterval(poll, 30000);
}

function animateConfidenceMeters() {
    document.querySelectorAll(".confidence-fill[data-progress]").forEach((bar) => {
        const progress = Number(bar.dataset.progress || 0);
        requestAnimationFrame(() => {
            bar.style.width = `${Math.max(0, Math.min(100, progress))}%`;
        });
    });

    document.querySelectorAll("[data-gauge-progress]").forEach((circle) => {
        const progress = Number(circle.dataset.gaugeProgress || 0);
        const radius = Number(circle.getAttribute("r") || 54);
        const circumference = 2 * Math.PI * radius;
        circle.style.strokeDasharray = `${circumference}`;
        circle.style.strokeDashoffset = `${circumference}`;
        requestAnimationFrame(() => {
            circle.style.strokeDashoffset = `${circumference - (circumference * Math.max(0, Math.min(100, progress))) / 100}`;
        });
    });
}

function bindToasts() {
    document.querySelectorAll(".toast").forEach((toast) => {
        window.setTimeout(() => toast.remove(), 4400);
    });
}

function bindPrintButtons() {
    ["printReportButton", "printReportButtonSecondary"].forEach((id) => {
        const button = document.getElementById(id);
        if (button) button.addEventListener("click", () => window.print());
    });
}

function bindRipples() {
    document.addEventListener("click", (event) => {
        const button = event.target.closest(".btn, .page-link");
        if (!button) return;
        const rect = button.getBoundingClientRect();
        const ripple = document.createElement("span");
        const size = Math.max(rect.width, rect.height);
        ripple.className = "ripple";
        ripple.style.width = `${size}px`;
        ripple.style.height = `${size}px`;
        ripple.style.left = `${event.clientX - rect.left - size / 2}px`;
        ripple.style.top = `${event.clientY - rect.top - size / 2}px`;
        button.appendChild(ripple);
        ripple.addEventListener("animationend", () => ripple.remove());
    });
}

function bindCursorGlow() {
    const glow = document.querySelector(".cursor-glow");
    if (!glow) return;

    document.addEventListener(
        "pointermove",
        throttle((event) => {
            glow.style.left = `${event.clientX}px`;
            glow.style.top = `${event.clientY}px`;
        }, 16),
        { passive: true }
    );
}

function bindCounters() {
    const counters = document.querySelectorAll("[data-count-to]");
    if (!counters.length) return;

    const animate = (el) => {
        const target = Number(el.dataset.countTo || 0);
        const suffix = el.dataset.countSuffix || "";
        const duration = 1100;
        const start = performance.now();

        const step = (now) => {
            const progress = Math.min(1, (now - start) / duration);
            const eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = `${Math.round(target * eased).toLocaleString()}${suffix}`;
            if (progress < 1) requestAnimationFrame(step);
        };

        requestAnimationFrame(step);
    };

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;
                animate(entry.target);
                observer.unobserve(entry.target);
            });
        },
        { threshold: 0.4 }
    );

    counters.forEach((counter) => observer.observe(counter));
}

function bindFloatingLabels() {
    document.querySelectorAll(".field.floating input, .field.floating select, .field.floating textarea").forEach((input) => {
        const field = input.closest(".field");
        const update = () => field?.classList.toggle("has-value", Boolean(input.value));
        input.addEventListener("input", update);
        input.addEventListener("change", update);
        update();
    });
}

function bindHistoryViewToggle() {
    const gridButton = document.querySelector("[data-history-view='grid']");
    const listButton = document.querySelector("[data-history-view='list']");
    const history = document.querySelector("[data-history-container]");
    if (!gridButton || !listButton || !history) return;

    const setView = (view) => {
        history.dataset.view = view;
        gridButton.classList.toggle("active", view === "grid");
        listButton.classList.toggle("active", view === "list");
        localStorage.setItem("anemiavision-history-view", view);
    };

    gridButton.addEventListener("click", () => setView("grid"));
    listButton.addEventListener("click", () => setView("list"));
    setView(localStorage.getItem("anemiavision-history-view") || "grid");
}

function bindDebouncedFilters() {
    const form = document.querySelector("[data-live-filter-form]");
    const search = form?.querySelector("input[type='search']");
    if (!form || !search) return;
    search.addEventListener("input", debounce(() => form.requestSubmit(), 300));
}

function revealSkeletons() {
    window.setTimeout(() => {
        document.querySelectorAll("[data-chart-shell]").forEach((shell) => shell.classList.remove("is-loading"));
    }, 800);
}

document.addEventListener("DOMContentLoaded", () => {
    document.body.classList.add("page-transition");
    window.setTimeout(() => document.body.classList.add("page-loaded"), 100);
    updateCurrentDateTime();
    window.setInterval(updateCurrentDateTime, 30000);
    bindNavbar();
    bindLiveStatus();
    animateConfidenceMeters();
    bindToasts();
    bindPrintButtons();
    bindRipples();
    bindCursorGlow();
    bindCounters();
    bindFloatingLabels();
    bindHistoryViewToggle();
    bindDebouncedFilters();
    revealSkeletons();

    if (window.lucide) {
        window.lucide.createIcons();
    }
});
