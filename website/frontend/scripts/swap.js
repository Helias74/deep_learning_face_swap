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
  
  if (resultZone) return resultZone;
  
  const labels = document.querySelectorAll('.label');
  for (let label of labels) {
    if (label.textContent.trim() === 'Résultat') {
      resultZone = label.nextElementSibling;
      if (resultZone && resultZone.classList.contains('dropzone')) {
        return resultZone;
      }
    }
  }
  
  const allCards = document.querySelectorAll('.card');
  const uploadGrid = document.querySelector('.upload-grid');
  
  for (let card of allCards) {
    if (!uploadGrid.contains(card)) {
      const dropzone = card.querySelector('.dropzone');
      if (dropzone) return dropzone;
    }
  }
  
  return null;
}

async function performSwap() {
  const resultCard = getResultZone();
  
  if (!resultCard) {
    alert("Erreur : zone de résultat introuvable");
    return;
  }
  
  resultCard.innerHTML = `
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:#666;">
      <div style="width:40px; height:40px; border:4px solid #f3f3f3; border-top:4px solid #3498db; border-radius:50%; animation:spin 1s linear infinite;"></div>
      <p style="margin-top:16px; font-size:14px;">Face swap en cours...</p>
    </div>
    <style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); }}</style>
  `;
  
  try {
    const formData = new FormData();
    formData.append("source", sourceFile);
    formData.append("target", targetFile);
    
    const response = await fetch(`${API}/swap/process`, {
      method: "POST",
      body: formData
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: "Erreur serveur" }));
      throw new Error(errorData.detail || `Erreur ${response.status}`);
    }
    
    const blob = await response.blob();
    const imageUrl = URL.createObjectURL(blob);
    
    resultCard.innerHTML = `
      <img src="${imageUrl}" style="width:100%; height:100%; object-fit:contain; border-radius:8px;">
      <button onclick="downloadImage('${imageUrl}')" style="position:absolute; bottom:16px; right:16px; padding:8px 16px; background:#3498db; color:white; border:none; border-radius:4px; cursor:pointer;">
        📥 Télécharger
      </button>
    `;
    
  } catch (error) {
    resultCard.innerHTML = `
      <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:#e74c3c;">
        <p style="font-size:18px;">❌ Erreur</p>
        <p style="font-size:14px; margin-top:8px;">${error.message}</p>
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