document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('messageInput');
    const scanBtn = document.getElementById('scanBtn');
    const btnText = document.querySelector('.btn-text');
    const iconBase = document.querySelector('.icon-default');
    const iconSpin = document.querySelector('.icon-scanning');
    
    const resultCard = document.getElementById('resultCard');
    const statusIcon = document.getElementById('statusIcon');
    const statusTitle = document.getElementById('statusTitle');
    const confidenceValue = document.getElementById('confidenceValue');
    const progressFill = document.getElementById('progressFill');
    const scamProb = document.getElementById('scamProb');
    const legitProb = document.getElementById('legitProb');

    const API_URL = 'http://localhost:8000/predict'; // Backend Endpoint

    scanBtn.addEventListener('click', async () => {
        const text = messageInput.value.trim();
        
        if (!text) {
            messageInput.style.borderColor = 'var(--danger)';
            setTimeout(() => messageInput.style.borderColor = 'var(--border-color)', 1000);
            return;
        }

        // Set Loading State
        scanBtn.disabled = true;
        btnText.textContent = "Analyzing...";
        iconBase.classList.add('hidden');
        iconSpin.classList.remove('hidden');
        resultCard.classList.add('hidden');
        progressFill.style.width = '0%';

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error('API server is not responding');
            }

            const data = await response.json();
            displayResult(data);

        } catch (error) {
            console.error("Error connecting to to API:", error);
            alert("Could not connect to the Backend API. Make sure 'python scam_detector.py --mode api' is running.");
        } finally {
            // Reset Button State
            scanBtn.disabled = false;
            btnText.textContent = "Initiate Scan";
            iconBase.classList.remove('hidden');
            iconSpin.classList.add('hidden');
        }
    });

    function displayResult(data) {
        resultCard.classList.remove('hidden', 'is-safe', 'is-scam');
        
        // Ensure values exist and fix precision 
        const isScam = data.is_scam;
        const confidence = data.confidence.toFixed(1);
        const scamProbability = data.scam_prob.toFixed(1);
        const legitProbability = data.legit_prob.toFixed(1);
        
        if (isScam) {
            resultCard.classList.add('is-scam');
            statusIcon.className = 'fa-solid fa-triangle-exclamation';
            statusTitle.textContent = 'Phishing / Scam Detected';
        } else {
            resultCard.classList.add('is-safe');
            statusIcon.className = 'fa-solid fa-shield-check';
            statusTitle.textContent = 'Message Appears Safe';
        }

        // Animate count up logic could be added here, but direct bind is fine
        confidenceValue.textContent = `${confidence}%`;
        scamProb.textContent = `${scamProbability}%`;
        legitProb.textContent = `${legitProbability}%`;

        // Animate Progress Bar slightly delayed
        setTimeout(() => {
            progressFill.style.width = `${confidence}%`;
        }, 100);
    }
});
