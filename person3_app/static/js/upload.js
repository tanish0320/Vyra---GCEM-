const MODALITY_GUIDANCE = {
    conjunctiva:
        "<strong>Conjunctiva capture guidance:</strong> Gently pull down the lower eyelid to expose the palpebral conjunctiva. Use bright, even lighting, avoid flash glare, and keep the camera steady (minimum 100×100 px).",
    fingernail:
        "<strong>Fingernail capture guidance:</strong> Photograph a clean, unpolished fingernail bed against a neutral surface under diffused light. Avoid pressing down on the nail tip, which artificially blanches the capillary bed (minimum 100×100 px).",
};

const SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"];
const SUPPORTED_MIME_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];

let currentImageRejected = false;
let currentRejectionReason = "";

function showValidationBanner(title, detail) {
    const banner = document.getElementById("scanValidationBanner");
    const titleEl = document.getElementById("scanValidationTitle");
    const detailEl = document.getElementById("scanValidationDetail");
    if (!banner || !titleEl || !detailEl) return;
    titleEl.textContent = title;
    detailEl.textContent = detail;
    banner.classList.remove("d-none");
}

function hideValidationBanner() {
    const banner = document.getElementById("scanValidationBanner");
    if (!banner) return;
    banner.classList.add("d-none");
}

function analyzeImageQuality(imageElement) {
    const natWidth = imageElement.naturalWidth || 0;
    const natHeight = imageElement.naturalHeight || 0;

    if (natWidth < 100 || natHeight < 100) {
        return {
            label: "Image Rejected",
            detail: `Resolution ${natWidth}×${natHeight} px is below the minimum 100×100 px required for screening.`,
            dimensions: `${natWidth}×${natHeight} px`,
            isGood: false,
            isRejected: true,
            state: "rejected",
        };
    }

    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d", { willReadFrequently: true });
    const width = 240;
    const height = Math.max(1, Math.round((natHeight / natWidth) * width));
    canvas.width = width;
    canvas.height = height;
    context.drawImage(imageElement, 0, 0, width, height);

    const imageData = context.getImageData(0, 0, width, height).data;
    const grayscale = new Float32Array(width * height);
    let brightnessTotal = 0;

    for (let index = 0; index < width * height; index += 1) {
        const offset = index * 4;
        const gray = 0.299 * imageData[offset] + 0.587 * imageData[offset + 1] + 0.114 * imageData[offset + 2];
        grayscale[index] = gray;
        brightnessTotal += gray;
    }

    const laplacianValues = [];
    for (let y = 1; y < height - 1; y += 1) {
        for (let x = 1; x < width - 1; x += 1) {
            const center = grayscale[y * width + x];
            const top = grayscale[(y - 1) * width + x];
            const bottom = grayscale[(y + 1) * width + x];
            const left = grayscale[y * width + (x - 1)];
            const right = grayscale[y * width + (x + 1)];
            laplacianValues.push(Math.abs(top + bottom + left + right - 4 * center));
        }
    }

    const mean = laplacianValues.length
        ? laplacianValues.reduce((sum, value) => sum + value, 0) / laplacianValues.length
        : 0;
    const variance = laplacianValues.length
        ? laplacianValues.reduce((sum, value) => sum + (value - mean) ** 2, 0) / laplacianValues.length
        : 0;
    const brightness = brightnessTotal / grayscale.length;

    if (brightness < 40) {
        return {
            label: "Poor lighting",
            detail: `Brightness ${brightness.toFixed(1)} is low. Retake under brighter, even lighting if possible.`,
            dimensions: `${natWidth}×${natHeight} px`,
            isGood: false,
            isRejected: false,
            state: "warning",
        };
    }
    if (variance < 100) {
        return {
            label: "Blurry image",
            detail: `Focus score ${variance.toFixed(1)} is soft. A sharper photo will improve screening reliability.`,
            dimensions: `${natWidth}×${natHeight} px`,
            isGood: false,
            isRejected: false,
            state: "warning",
        };
    }
    return {
        label: "Good quality",
        detail: `Focus score ${variance.toFixed(1)} · Brightness ${brightness.toFixed(0)}. Suitable for screening.`,
        dimensions: `${natWidth}×${natHeight} px`,
        isGood: true,
        isRejected: false,
        state: "good",
    };
}

function updateQualityStatus(analysis) {
    const qualityStatus = document.getElementById("qualityStatus");
    const qualityScore = document.getElementById("qualityScore");
    const qualityDimensions = document.getElementById("qualityDimensions");
    if (!qualityStatus || !qualityScore) return;

    qualityStatus.textContent = analysis.label;
    qualityStatus.dataset.qualityState = analysis.state || (analysis.isGood ? "good" : "warning");
    qualityStatus.classList.toggle("text-success", Boolean(analysis.isGood));
    qualityStatus.classList.toggle("text-danger", !analysis.isGood);
    qualityScore.textContent = analysis.detail;
    if (qualityDimensions) {
        qualityDimensions.textContent = analysis.dimensions || "";
    }

    currentImageRejected = Boolean(analysis.isRejected);
    currentRejectionReason = analysis.isRejected ? analysis.detail : "";
    if (analysis.isRejected) {
        showValidationBanner("Image Rejected", analysis.detail);
    } else {
        hideValidationBanner();
    }
}

function showPreviewFromDataUrl(dataUrl) {
    const preview = document.getElementById("imagePreview");
    const placeholder = document.getElementById("previewPlaceholder");
    if (!preview || !placeholder) return;
    preview.src = dataUrl;
    preview.classList.remove("d-none");
    placeholder.classList.add("d-none");
    preview.onload = () => updateQualityStatus(analyzeImageQuality(preview));
}

function resetPreview() {
    const preview = document.getElementById("imagePreview");
    const placeholder = document.getElementById("previewPlaceholder");
    const qualityStatus = document.getElementById("qualityStatus");
    const qualityScore = document.getElementById("qualityScore");
    const qualityDimensions = document.getElementById("qualityDimensions");
    if (!preview || !placeholder) return;
    preview.src = "";
    preview.classList.add("d-none");
    placeholder.classList.remove("d-none");
    currentImageRejected = false;
    currentRejectionReason = "";
    if (qualityStatus) {
        qualityStatus.textContent = "Awaiting image";
        qualityStatus.dataset.qualityState = "awaiting";
        qualityStatus.classList.remove("text-success", "text-danger");
    }
    if (qualityScore) {
        qualityScore.textContent = "Upload or capture an image to run client-side resolution, lighting, and focus checks.";
    }
    if (qualityDimensions) {
        qualityDimensions.textContent = "";
    }
}

function setCameraStatus(message, isError = false) {
    const status = document.getElementById("cameraStatus");
    if (!status) return;
    status.textContent = message;
    status.classList.toggle("text-danger", isError);
}

function isLocalHost() {
    return ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
}

function shouldWarnAboutHttps() {
    return !window.isSecureContext && !isLocalHost();
}

function toggleCameraWarning(show) {
    document.getElementById("cameraHttpWarning")?.classList.toggle("d-none", !show);
}

function syncCameraHud(modalityKey) {
    const isFingernail = modalityKey === "fingernail" || modalityKey === "Fingernail";
    const eyeGuide = document.getElementById("reticleEyeGuide");
    const nailGuide = document.getElementById("reticleNailGuide");
    const modalityLabel = document.getElementById("hudModalityLabel");
    const instruction = document.getElementById("hudInstruction");

    if (eyeGuide && nailGuide) {
        eyeGuide.classList.toggle("d-none", isFingernail);
        nailGuide.classList.toggle("d-none", !isFingernail);
    }
    if (modalityLabel) {
        modalityLabel.textContent = isFingernail ? "Fingernail Bed" : "Eye Conjunctiva";
    }
    if (instruction) {
        instruction.textContent = isFingernail
            ? "Align fingernail bed inside green box & hold steady"
            : "Align lower eyelid inside green guide & pull down";
    }
}

function bindModalitySelector() {
    const cards = Array.from(document.querySelectorAll(".modality-card"));
    const select = document.getElementById("imageType");
    const bodyPartInput = document.getElementById("bodyPartInput");
    const guidanceText = document.getElementById("modalityGuidanceText");

    const syncModality = (modalityKey) => {
        const normalized = modalityKey === "fingernail" || modalityKey === "Fingernail" ? "fingernail" : "conjunctiva";
        const uiLabel = normalized === "fingernail" ? "Fingernail" : "Eye Conjunctiva";

        cards.forEach((card) => {
            const isMatch = card.dataset.modality === normalized;
            card.classList.toggle("active", isMatch);
            card.setAttribute("aria-pressed", isMatch ? "true" : "false");
        });

        if (select && select.value !== uiLabel) {
            select.value = uiLabel;
        }
        if (bodyPartInput) {
            bodyPartInput.value = normalized;
        }
        if (guidanceText) {
            guidanceText.innerHTML = MODALITY_GUIDANCE[normalized];
        }
        syncCameraHud(normalized);
    };

    cards.forEach((card) => {
        card.addEventListener("click", () => {
            syncModality(card.dataset.modality || "conjunctiva");
        });
    });

    select?.addEventListener("change", () => {
        syncModality(select.value);
    });
}

function bindTabs() {
    document.querySelectorAll(".scan-tab").forEach((tab) => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".scan-tab").forEach((item) => {
                item.classList.remove("active");
                item.setAttribute("aria-selected", "false");
            });
            document.querySelectorAll(".tab-pane").forEach((pane) => pane.classList.remove("show", "active"));
            tab.classList.add("active");
            tab.setAttribute("aria-selected", "true");
            const target = document.querySelector(tab.dataset.target || "");
            target?.classList.add("show", "active");
        });
    });
}

function isSupportedImageFile(file) {
    if (!file) return false;
    const name = (file.name || "").toLowerCase();
    const type = (file.type || "").toLowerCase();
    const hasValidExt = SUPPORTED_EXTENSIONS.some((ext) => name.endsWith(ext));
    const hasValidMime = SUPPORTED_MIME_TYPES.includes(type);
    return hasValidExt || hasValidMime;
}

function bindUploadZone() {
    const dropzone = document.getElementById("dropzone");
    const imageInput = document.getElementById("imageInput");
    const capturedInput = document.getElementById("capturedImageData");
    const uploadTabButton = document.getElementById("upload-tab");
    const fileMeta = document.getElementById("fileMeta");
    if (!dropzone || !imageInput) return;

    const readFile = (file) => {
        if (!isSupportedImageFile(file)) {
            resetPreview();
            currentImageRejected = true;
            currentRejectionReason = "Unsupported file type. Please upload a JPG, JPEG, PNG, or WEBP image.";
            updateQualityStatus({
                label: "Image Rejected",
                detail: currentRejectionReason,
                dimensions: "",
                isGood: false,
                isRejected: true,
                state: "rejected",
            });
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            resetPreview();
            currentImageRejected = true;
            currentRejectionReason = "The uploaded image exceeds the maximum allowed size of 10 MB.";
            updateQualityStatus({
                label: "Image Rejected",
                detail: currentRejectionReason,
                dimensions: "",
                isGood: false,
                isRejected: true,
                state: "rejected",
            });
            return;
        }

        const reader = new FileReader();
        reader.onload = (event) => {
            if (capturedInput) capturedInput.value = "";
            showPreviewFromDataUrl(event.target?.result || "");
            if (fileMeta) fileMeta.textContent = `${file.name} - ${(file.size / 1024 / 1024).toFixed(2)} MB`;
        };
        reader.readAsDataURL(file);
    };

    imageInput.addEventListener("change", () => {
        const file = imageInput.files?.[0];
        if (file) readFile(file);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (event) => {
        const file = event.dataTransfer?.files?.[0];
        if (file) {
            imageInput.files = event.dataTransfer.files;
            readFile(file);
        }
    });

    uploadTabButton?.addEventListener("click", () => toggleCameraWarning(false));

    // Secondary Image Input & Dual-Scan Toggle wiring
    const dualCheckbox = document.getElementById("dualScanModeCheckbox");
    const secondaryZone = document.getElementById("secondaryScanZone");
    const secondaryInput = document.getElementById("secondaryImageInput");
    const secondaryFileMeta = document.getElementById("secondaryFileMeta");

    dualCheckbox?.addEventListener("change", () => {
        secondaryZone?.classList.toggle("d-none", !dualCheckbox.checked);
        if (window.lucide) window.lucide.createIcons();
    });

    secondaryInput?.addEventListener("change", () => {
        const file = secondaryInput.files?.[0];
        if (file && secondaryFileMeta) {
            secondaryFileMeta.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB) · Ready for Dual-Site Fusion`;
        }
    });
}

function bindCameraCapture() {
    const startButton = document.getElementById("startCameraButton");
    const captureButton = document.getElementById("captureButton");
    const retakeButton = document.getElementById("retakeButton");
    const confirmButton = document.getElementById("confirmCaptureButton");
    const video = document.getElementById("cameraFeed");
    const canvas = document.getElementById("cameraCanvas");
    const capturedInput = document.getElementById("capturedImageData");
    const imageInput = document.getElementById("imageInput");
    const uploadTabButton = document.getElementById("upload-tab");
    const cameraTabButton = document.getElementById("camera-tab");
    let stream = null;
    if (!startButton || !captureButton || !video || !canvas || !capturedInput) return;

    const toggleCaptureState = (hasCapture) => {
        retakeButton?.classList.toggle("d-none", !hasCapture);
        confirmButton?.classList.toggle("d-none", !hasCapture);
        captureButton.classList.toggle("d-none", hasCapture);
    };

    let qualityInterval = null;
    const hudQualityText = document.getElementById("hudQualityText");
    const hudQualityPill = document.getElementById("hudQualityPill");
    const scanLaser = document.getElementById("reticleScanLaser");

    const startLiveAnalysis = () => {
        if (qualityInterval) clearInterval(qualityInterval);
        const miniCanvas = document.createElement("canvas");
        miniCanvas.width = 64;
        miniCanvas.height = 48;
        const miniCtx = miniCanvas.getContext("2d", { willReadFrequently: true });

        qualityInterval = setInterval(() => {
            if (!video.videoWidth || !video.videoHeight || video.paused) return;
            try {
                // Sample center 40% of video frame (the reticle zone)
                const sx = Math.floor(video.videoWidth * 0.3);
                const sy = Math.floor(video.videoHeight * 0.3);
                const sw = Math.floor(video.videoWidth * 0.4);
                const sh = Math.floor(video.videoHeight * 0.4);
                miniCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 64, 48);
                const p = miniCtx.getImageData(0, 0, 64, 48).data;
                let bSum = 0;
                for (let i = 0; i < p.length; i += 4) {
                    bSum += 0.299 * p[i] + 0.587 * p[i + 1] + 0.114 * p[i + 2];
                }
                const avgB = bSum / (p.length / 4);

                if (hudQualityText && hudQualityPill) {
                    if (avgB < 48) {
                        hudQualityText.textContent = "Low Light";
                        hudQualityPill.className = "hud-pill hud-pill--status text-warning";
                    } else if (avgB > 218) {
                        hudQualityText.textContent = "Glare Warning";
                        hudQualityPill.className = "hud-pill hud-pill--status text-warning";
                    } else {
                        hudQualityText.textContent = "Optimal Frame";
                        hudQualityPill.className = "hud-pill hud-pill--status";
                    }
                }
            } catch (err) {
                // silent fallback
            }
        }, 280);
    };

    const stopStreamTracks = () => {
        if (qualityInterval) {
            clearInterval(qualityInterval);
            qualityInterval = null;
        }
        if (hudQualityText) hudQualityText.textContent = "Awaiting Camera";
        if (!stream) return;
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
        video.srcObject = null;
    };

    startButton.addEventListener("click", async () => {
        if (shouldWarnAboutHttps()) {
            toggleCameraWarning(true);
            setCameraStatus("Camera access needs HTTPS. Use file upload or open the secure site.", true);
            uploadTabButton?.click();
            return;
        }

        if (!navigator.mediaDevices?.getUserMedia) {
            setCameraStatus("Camera access is not available on this device. Please use file upload.", true);
            uploadTabButton?.click();
            return;
        }

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
                audio: false,
            });
            video.srcObject = stream;
            setCameraStatus("Camera ready. Align anatomical target with green reticle, then tap Capture.");
            video.onloadedmetadata = () => {
                video.play();
                startLiveAnalysis();
            };
        } catch (error) {
            setCameraStatus("Unable to access the camera. Allow permission or switch to file upload.", true);
            uploadTabButton?.click();
        }
    });

    captureButton.addEventListener("click", () => {
        if (!video.videoWidth || !video.videoHeight) {
            setCameraStatus("Start the camera first, then capture the image.", true);
            return;
        }
        if (scanLaser) {
            scanLaser.classList.add("scanning");
            setTimeout(() => scanLaser.classList.remove("scanning"), 700);
        }
        const context = canvas.getContext("2d");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
        capturedInput.value = dataUrl;
        if (imageInput) imageInput.value = "";
        showPreviewFromDataUrl(dataUrl);
        toggleCaptureState(true);
        setCameraStatus("Photo captured & locked. Review quality indicators below or tap Retake.");
    });

    retakeButton?.addEventListener("click", () => {
        capturedInput.value = "";
        toggleCaptureState(false);
        resetPreview();
        hideValidationBanner();
        setCameraStatus("Retake the photo when the image is sharp and evenly lit.");
    });

    confirmButton?.addEventListener("click", () => setCameraStatus("Captured image confirmed and ready for AI screening."));
    window.addEventListener("beforeunload", stopStreamTracks);
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) stopStreamTracks();
    });
    cameraTabButton?.addEventListener("click", () => toggleCameraWarning(shouldWarnAboutHttps()));
    uploadTabButton?.addEventListener("click", () => {
        stopStreamTracks();
        toggleCameraWarning(false);
    });
}

function bindScanSubmit() {
    const form = document.getElementById("scanForm");
    const overlay = document.getElementById("loadingOverlay");
    const imageInput = document.getElementById("imageInput");
    const capturedInput = document.getElementById("capturedImageData");
    const submitButton = document.getElementById("submitScanButton");
    const patientName = document.getElementById("patientName");
    const patientAge = document.getElementById("patientAge");
    const gender = document.getElementById("gender");
    if (!form || !overlay) return;

    const initialDisabledState = Boolean(submitButton?.disabled);
    const initialLabel = submitButton?.innerHTML || "";

    const restoreSubmitState = () => {
        overlay.classList.remove("show");
        overlay.setAttribute("aria-hidden", "true");
        if (submitButton) {
            submitButton.disabled = initialDisabledState;
            submitButton.innerHTML = initialLabel;
        }
        if (window.lucide) window.lucide.createIcons();
    };

    form.addEventListener("submit", (event) => {
        const hasUpload = Boolean(imageInput?.files?.length);
        const hasCapture = Boolean(capturedInput?.value);

        if (!hasUpload && !hasCapture) {
            event.preventDefault();
            showValidationBanner("Image Rejected", "Please upload an image or capture one from the camera before submitting.");
            return;
        }

        if (currentImageRejected) {
            event.preventDefault();
            showValidationBanner("Image Rejected", currentRejectionReason || "The selected image did not pass validation checks.");
            return;
        }

        const nameVal = (patientName?.value || "").trim();
        const ageRaw = (patientAge?.value || "").trim();
        const ageNum = Number(ageRaw);
        const genderVal = (gender?.value || "").trim();

        if (!nameVal) {
            event.preventDefault();
            showValidationBanner("Processing Error · Missing Patient Intake", "Patient name is required before running screening.");
            patientName?.focus();
            return;
        }
        if (ageRaw === "" || Number.isNaN(ageNum) || !Number.isInteger(ageNum) || ageNum < 0 || ageNum > 120) {
            event.preventDefault();
            showValidationBanner("Processing Error · Invalid Age", "Please enter a valid patient age between 0 and 120 years.");
            patientAge?.focus();
            return;
        }
        if (!genderVal) {
            event.preventDefault();
            showValidationBanner("Processing Error · Missing Gender", "Please select the patient's biological sex / gender.");
            gender?.focus();
            return;
        }

        hideValidationBanner();
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.innerHTML = '<i data-lucide="dna"></i><span>Submitting to Server...</span>';
            if (window.lucide) window.lucide.createIcons();
        }
        overlay.classList.add("show");
        overlay.setAttribute("aria-hidden", "false");
    });

    window.addEventListener("pageshow", restoreSubmitState);
}

function bindModelReadiness() {
    const submitButton = document.getElementById("submitScanButton");
    const badge = document.getElementById("modelStatusBadge");
    const hint = document.getElementById("modelStatusHint");
    const chip = document.getElementById("modelStateChip");
    const stagePreprocessStatus = document.getElementById("stagePreprocessStatus");
    const stagePredictStatus = document.getElementById("stagePredictStatus");
    if (!submitButton || !badge || !hint) return;

    const card = badge.closest(".model-readiness-card");

    const setReadyState = (payload) => {
        submitButton.disabled = false;
        badge.textContent = "Model Ready";
        hint.textContent = "The AI model is loaded on this server and ready for screening.";
        if (chip) {
            chip.textContent = "Ready for Inference";
            chip.className = "state-chip state-chip--ready";
        }
        if (card) {
            card.dataset.modelState = "ready";
            card.classList.add("model-readiness-card--ready");
            card.classList.remove("model-readiness-card--loading", "model-readiness-card--offline");
        }
        if (stagePreprocessStatus) {
            stagePreprocessStatus.textContent = payload?.person1_normalizer_available
                ? "Person 1 optical normalization module detected on server."
                : "Person 1 normalizer not installed; server will pass validated raw image without fabricated normalization.";
        }
        if (stagePredictStatus) {
            stagePredictStatus.textContent = "Model ready on server; prediction will execute after upload validation.";
        }
    };

    const setUnavailableState = (reasonText, payload) => {
        submitButton.disabled = false;
        badge.textContent = "Model Unavailable";
        hint.textContent = reasonText || "A valid trained checkpoint is not available on this server.";
        if (chip) {
            chip.textContent = payload?.checkpoint_is_lfs_pointer ? "Git LFS Pointer Detected" : "Weights Missing";
            chip.className = "state-chip state-chip--unavailable";
        }
        if (card) {
            card.dataset.modelState = "unavailable";
            card.classList.add("model-readiness-card--offline");
            card.classList.remove("model-readiness-card--loading", "model-readiness-card--ready");
        }
        if (stagePreprocessStatus) {
            stagePreprocessStatus.textContent = payload?.person1_normalizer_available
                ? "Person 1 optical normalization module detected on server."
                : "Person 1 normalizer not installed on server.";
        }
        if (stagePredictStatus) {
            stagePredictStatus.textContent = "Live AI model weights are currently unavailable on server.";
        }
    };

    let attempts = 0;
    const poll = async () => {
        attempts += 1;
        try {
            const response = await fetch("/health", { headers: { Accept: "application/json" }, cache: "no-store" });
            if (!response.ok) {
                setUnavailableState("Health check returned an error; server will validate readiness on submit.");
                return true;
            }
            const payload = await response.json();
            if (payload.predictor_loaded || payload.model_ready) {
                setReadyState(payload);
                return true;
            }
            if (
                !payload.checkpoint_available ||
                payload.checkpoint_valid === false ||
                payload.checkpoint_is_lfs_pointer ||
                payload.failure_category ||
                payload.predictor_load_error
            ) {
                setUnavailableState(
                    payload.status_message ||
                        payload.predictor_load_error ||
                        "Model weights are unavailable on this deployment.",
                    payload
                );
                return true;
            }
        } catch (error) {
            if (attempts >= 3) {
                setUnavailableState("Could not reach /health endpoint; you may still submit to test server readiness.");
                return true;
            }
        }
        if (attempts >= 30) {
            setUnavailableState("Model warmup timed out; submit the form to check live server status.");
            return true;
        }
        return false;
    };

    poll().then((done) => {
        if (done) return;
        const timer = window.setInterval(async () => {
            if (await poll()) window.clearInterval(timer);
        }, 2000);
    });
}

document.addEventListener("DOMContentLoaded", () => {
    bindModalitySelector();
    bindTabs();
    bindUploadZone();
    bindCameraCapture();
    bindScanSubmit();
    bindModelReadiness();
});
