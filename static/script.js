// ----- Image preview -----
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const newsInput = document.getElementById("newsInput");
const newsUrl = document.getElementById("newsUrl");
const checkBtn = document.getElementById("checkBtn");
const statusMessage = document.getElementById("statusMessage");
const resultBlock = document.getElementById("result");

const resultSummary = document.getElementById("resultSummary");
const modelLabel = document.getElementById("modelLabel");
const modelConfidence = document.getElementById("modelConfidence");
const verificationMessage = document.getElementById("verificationMessage");
const verificationCounts = document.getElementById("verificationCounts");
const sourcesList = document.getElementById("sourcesList");
const searchResultsList = document.getElementById("searchResultsList");

if (imageInput) {
    imageInput.addEventListener("change", function () {
        const file = this.files[0];
        if (file) {
            preview.src = URL.createObjectURL(file);
            preview.style.display = "block";
        } else {
            preview.style.display = "none";
        }
    });
}

// Utility to clear lists
function clearList(element) {
    while (element.firstChild) {
        element.removeChild(element.firstChild);
    }
}

// ----- Check button -----
if (checkBtn) {
    checkBtn.addEventListener("click", async () => {
        const text = newsInput.value.trim();
        const url = newsUrl.value.trim();
        const imageFile = imageInput.files[0];

        if (!text && !url && !imageFile) {
            statusMessage.textContent = "Please enter text, add a URL, or upload an image.";
            statusMessage.className = "status-message error";
            resultBlock.style.display = "none";
            return;
        }

        statusMessage.textContent = "Checking news... please wait ⏳";
        statusMessage.className = "status-message loading";
        resultBlock.style.display = "none";

        const formData = new FormData();
        formData.append("text", text);
        formData.append("url", url);
        if (imageFile) {
            formData.append("image", imageFile);
        }

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                const msg = data.error || "Server returned an error.";
                statusMessage.textContent = "⚠️ " + msg;
                statusMessage.className = "status-message error";
                resultBlock.style.display = "none";
                return;
            }

            // Populate results
            statusMessage.textContent = "Analysis complete ✅";
            statusMessage.className = "status-message success";
            resultBlock.style.display = "block";

            const label = data.model?.label || "UNKNOWN";
            const confidence = data.model?.confidence ?? null;
            const verification = data.verification || {};
            const searchResults = data.search_results || [];
            const trustedExamples = verification.trusted_examples || [];
            const untrustedExamples = verification.untrusted_examples || [];

            // Summary
            resultSummary.textContent = data.input_preview
                ? "Text used for analysis (preview): " + data.input_preview
                : "Analysis completed based on the provided inputs.";

            // Model
            modelLabel.textContent = label === "REAL"
                ? "🟢 Model thinks this is: REAL"
                : label === "FAKE"
                    ? "🔴 Model thinks this is: FAKE"
                    : "Model label: " + label;

            if (confidence !== null) {
                modelConfidence.textContent = "Confidence: " + (confidence * 100).toFixed(1) + "%";
            } else {
                modelConfidence.textContent = "";
            }

            // Verification summary
            verificationMessage.textContent = verification.message || "No verification summary available.";
            const trustedCount = verification.trusted_count ?? 0;
            const untrustedCount = verification.untrusted_count ?? 0;
            verificationCounts.textContent = `Trusted sources: ${trustedCount} | Other sources: ${untrustedCount}`;

            // Top sources list
            clearList(sourcesList);
            const topSources = [...trustedExamples, ...untrustedExamples].slice(0, 5);

            if (topSources.length === 0) {
                const li = document.createElement("li");
                li.textContent = "No specific sources found.";
                sourcesList.appendChild(li);
            } else {
                for (const item of topSources) {
                    const li = document.createElement("li");
                    const a = document.createElement("a");
                    a.href = item.url;
                    a.target = "_blank";
                    a.rel = "noopener noreferrer";
                    const tag = item.is_trusted ? "✅ Trusted" : "⚠️ Unknown";
                    a.textContent = `[${tag}] ${item.source || "Source"} – ${item.title || "View article"}`;
                    li.appendChild(a);
                    sourcesList.appendChild(li);
                }
            }

            // Full search results list
            clearList(searchResultsList);
            if (searchResults.length === 0) {
                const li = document.createElement("li");
                li.textContent = "No related news found in search.";
                searchResultsList.appendChild(li);
            } else {
                for (const item of searchResults) {
                    const li = document.createElement("li");
                    const a = document.createElement("a");
                    a.href = item.url;
                    a.target = "_blank";
                    a.rel = "noopener noreferrer";
                    const tag = item.is_trusted ? "✅ Trusted" : "⚠️ Unknown";
                    a.textContent = `[${tag}] ${item.source || "Source"} – ${item.title || "View article"}`;
                    li.appendChild(a);
                    searchResultsList.appendChild(li);
                }
            }
        } catch (err) {
            console.error(err);
            statusMessage.textContent = "⚠️ Something went wrong while contacting the server.";
            statusMessage.className = "status-message error";
            resultBlock.style.display = "none";
        }
    });
}
