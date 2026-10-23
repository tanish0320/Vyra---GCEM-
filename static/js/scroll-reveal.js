(function initScrollReveal() {
    const revealItems = () => document.querySelectorAll(".reveal, .fade-in-up");

    if (!("IntersectionObserver" in window)) {
        revealItems().forEach((item) => item.classList.add("is-visible"));
        return;
    }

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("is-visible");
                    observer.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.12, rootMargin: "0px 0px -8% 0px" }
    );

    document.addEventListener("DOMContentLoaded", () => {
        revealItems().forEach((item, index) => {
            item.classList.add("reveal");
            item.style.transitionDelay = `${Math.min(index % 8, 6) * 70}ms`;
            observer.observe(item);
        });
    });
})();
