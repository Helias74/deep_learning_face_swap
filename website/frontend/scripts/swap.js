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
  
  if (sourceFile && targetFile) performCrop();
});

document.getElementById("input-target").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  
  targetFile = file;
  document.getElementById("preview-target").src = URL.createObjectURL(file);
  document.getElementById("preview-target").style.display = "block";
  document.getElementById("text-target").style.display = "none";
  
  if (sourceFile && targetFile) performCrop();
});

async function performCrop() {
  const resultCard = document.querySelector('.card:last-child .dropzone');
  
  resultCard.innerHTML = `
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:#666;">
      <div style="width:40px; height:40px; border:4px solid #f3f3f3; border-top:4px solid #3498db; border-radius:50%; animation:spin 1s linear infinite;"></div>
      <p style="margin-top:16px; font-size:14px;">Crop en cours...</p>
    </div>
    <style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); }}</style>
  `;
  
  try {
    const formData = new FormData();
    formData.append("source", sourceFile);
    formData.append("target", targetFile);
    
    const response = await fetch(`${API}/swap/crop`, { method: "POST", body: formData });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Erreur lors du crop");
    }
    
    const blob = await response.blob();
    const imageUrl = URL.createObjectURL(blob);
    
    resultCard.innerHTML = `
      <img src="${imageUrl}" style="width:100%; height:100%; object-fit:contain; border-radius:8px;">
      <p style="position:absolute; bottom:16px; left:16px; background:rgba(0,0,0,0.7); color:white; padding:8px 12px; border-radius:4px; font-size:12px;">
        Image source croppée ✓
      </p>
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