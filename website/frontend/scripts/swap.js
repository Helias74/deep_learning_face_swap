let sourceFile = null;
let targetFile = null;

document.getElementById("drop-source").onclick = () => 
  document.getElementById("input-source").click();

document.getElementById("drop-target").onclick = () => 
  document.getElementById("input-target").click();

document.getElementById("input-source").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  
  sourceFile = file;
  document.getElementById("preview-source").src = URL.createObjectURL(file);
  document.getElementById("preview-source").style.display = "block";
  document.getElementById("text-source").style.display = "none";
  
  if (sourceFile && targetFile) performSwap();
});

document.getElementById("input-target").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  
  targetFile = file;
  document.getElementById("preview-target").src = URL.createObjectURL(file);
  document.getElementById("preview-target").style.display = "block";
  document.getElementById("text-target").style.display = "none";
  
  if (sourceFile && targetFile) performSwap();
});

function getResultZone() {
  let resultZone = document.getElementById("result-zone");
  
  if (resultZone) {
    console.log("✅ Zone résultat trouvée par ID");
    return resultZone;
  }
  
  console.warn("⚠️ ID result-zone absent, recherche par label...");
  const labels = document.querySelectorAll('.label');
  
  for (let label of labels) {
    if (label.textContent.trim() === 'Résultat') {
      resultZone = label.nextElementSibling;
      if (resultZone && resultZone.classList.contains('dropzone')) {
        console.log("✅ Zone résultat trouvée par label");
        return resultZone;
      }
    }
  }
  
  console.warn("⚠️ Recherche par position...");
  const allCards = document.querySelectorAll('.card');
  const uploadGrid = document.querySelector('.upload-grid');
  
  for (let card of allCards) {
    if (!uploadGrid.contains(card)) {
      const dropzone = card.querySelector('.dropzone');
      if (dropzone) {
        console.log("✅ Zone résultat trouvée par position");
        return dropzone;
      }
    }
  }
  
  console.error("❌ Impossible de trouver la zone de résultat !");
  return null;
}

// ════════════════════════════════════════════════════════════
// FONCTION : Appel API avec retry et gestion du cold start
// ════════════════════════════════════════════════════════════
async function fetchWithRetry(url, options, maxRetries = 2) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`🔄 Tentative ${attempt}/${maxRetries}...`);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minutes timeout
      
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      return response;
      
    } catch (error) {
      console.error(`❌ Tentative ${attempt} échouée:`, error.message);
      
      if (attempt < maxRetries) {
        console.log(`🔄 Nouvelle tentative dans 3 secondes...`);
        await new Promise(resolve => setTimeout(resolve, 3000));
      } else {
        throw error;
      }
    }
  }
}

async function performSwap() {
  const resultCard = getResultZone();
  
  if (!resultCard) {
    console.error("❌ Zone de résultat introuvable");
    alert("Erreur : impossible de trouver la zone de résultat. Veuillez rafraîchir la page.");
    return;
  }
  
  resultCard.innerHTML = `
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:#666;">
      <div style="width:40px; height:40px; border:4px solid #f3f3f3; border-top:4px solid #3498db; border-radius:50%; animation:spin 1s linear infinite;"></div>
      <p style="margin-top:16px; font-size:14px;">Démarrage du serveur...</p>
      <p style="font-size:12px; color:#999; margin-top:8px;">Première utilisation : 30-60s</p>
      <p style="font-size:12px; color:#999; margin-top:4px;">Ensuite : Crop + Swap (30-60s)</p>
    </div>
    <style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); }}</style>
  `;
  
  try {
    const formData = new FormData();
    formData.append("source", sourceFile);
    formData.append("target", targetFile);
    
    // ════════════════════════════════════════════════════════════
    // Appel avec retry pour gérer le cold start
    // ════════════════════════════════════════════════════════════
    const response = await fetchWithRetry(`${API}/swap/process`, {
      method: "POST",
      body: formData
    }, 2);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: "Erreur serveur" }));
      throw new Error(errorData.detail || `Erreur ${response.status}`);
    }
    
    const blob = await response.blob();
    const imageUrl = URL.createObjectURL(blob);
    
    resultCard.innerHTML = `
      <img src="${imageUrl}" style="width:100%; height:100%; object-fit:contain; border-radius:8px;">
      <p style="position:absolute; bottom:16px; left:16px; background:rgba(0,0,0,0.7); color:white; padding:8px 12px; border-radius:4px; font-size:12px;">
        Face swap réussi ✓
      </p>
      <button onclick="downloadImage('${imageUrl}')" style="position:absolute; bottom:16px; right:16px; padding:8px 16px; background:#3498db; color:white; border:none; border-radius:4px; cursor:pointer;">
        📥 Télécharger
      </button>
    `;
    
  } catch (error) {
    console.error("❌ Erreur finale:", error);
    
    resultCard.innerHTML = `
      <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:#e74c3c;">
        <p style="font-size:18px;">❌ Erreur</p>
        <p style="font-size:14px; margin-top:8px;">${error.message}</p>
        <p style="font-size:12px; color:#999; margin-top:8px;">Le serveur gratuit peut mettre jusqu'à 60s à démarrer</p>
        <button onclick="location.reload()" style="margin-top:16px; padding:8px 16px; background:#3498db; color:white; border:none; border-radius:4px; cursor:pointer;">
          Réessayer
        </button>
      </div>
    `;
  }
}

function downloadImage(url) {
  const a = document.createElement('a');
  a.href = url;
  a.download = `faceswap_${Date.now()}.jpg`;
  a.click();
}


// ════════════════════════════════════════════════════════════
// Pré-charger les modèles au chargement de la page
// ════════════════════════════════════════════════════════════
window.addEventListener('DOMContentLoaded', async () => {
  console.log("🔄 Pré-chargement des modèles...");
  
  try {
    const response = await fetch(`${API}/swap/warmup`, { method: "POST" });
    const data = await response.json();
    
    if (data.ready) {
      console.log("✅ Modèles prêts !");
    } else {
      console.warn("⚠️ Modèles non chargés, premier swap sera lent");
    }
  } catch (error) {
    console.warn("⚠️ Erreur pré-chargement:", error);
  }
});