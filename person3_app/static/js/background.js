(function initBackground() {
    const canvas = document.getElementById("bg-canvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let width = 0;
    let height = 0;
    let particles = [];
    let mouseX = 0;
    let mouseY = 0;
    let frame = 0;

    function resize() {
        const ratio = Math.min(window.devicePixelRatio || 1, 2);
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = Math.floor(width * ratio);
        canvas.height = Math.floor(height * ratio);
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    }

    function createParticles() {
        const count = Math.min(100, Math.max(56, Math.floor((window.innerWidth * window.innerHeight) / 18000)));
        particles = Array.from({ length: count }, () => ({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.32,
            vy: (Math.random() - 0.5) * 0.32,
            r: 1 + Math.random() * 1.7
        }));
    }

    function draw() {
        ctx.clearRect(0, 0, width, height);
        ctx.save();
        ctx.translate((mouseX - width / 2) * 0.05, (mouseY - height / 2) * 0.05);

        for (const p of particles) {
            if (!reducedMotion) {
                p.x += p.vx;
                p.y += p.vy;
                if (p.x < -20) p.x = width + 20;
                if (p.x > width + 20) p.x = -20;
                if (p.y < -20) p.y = height + 20;
                if (p.y > height + 20) p.y = -20;
            }

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(148, 163, 184, 0.46)";
            ctx.fill();
        }

        for (let i = 0; i < particles.length; i += 1) {
            for (let j = i + 1; j < particles.length; j += 1) {
                const a = particles[i];
                const b = particles[j];
                const dx = a.x - b.x;
                const dy = a.y - b.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.strokeStyle = `rgba(99, 102, 241, ${0.18 * (1 - dist / 120)})`;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
        }

        ctx.restore();
        frame = requestAnimationFrame(draw);
    }

    window.addEventListener("resize", () => {
        resize();
        createParticles();
    });

    document.addEventListener(
        "pointermove",
        (event) => {
            mouseX = event.clientX;
            mouseY = event.clientY;
        },
        { passive: true }
    );

    document.addEventListener("visibilitychange", () => {
        if (document.hidden) {
            cancelAnimationFrame(frame);
            frame = 0;
        } else if (!frame) {
            draw();
        }
    });

    resize();
    createParticles();
    draw();
})();
