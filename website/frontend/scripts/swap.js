document.addEventListener("DOMContentLoaded", async () => {
  const res = await fetch("http://localhost:8000/swap/models");
  const models = await res.json();
  
  const select = document.getElementById("model-select");
  select.innerHTML = "";
  
  models.forEach(m => {
    const option = document.createElement("option");
    option.value = m.file_path;
    option.textContent = m.name;
    select.appendChild(option);
  });

  const selectedModel = select.value;
});





document.getElementById("drop-source").onclick = () => document.getElementById("input-source").click();
document.getElementById("drop-target").onclick = () => document.getElementById("input-target").click();

document.getElementById("input-source").addEventListener("change", (e) => {
  const file = e.target.files[0];
  document.getElementById("preview-source").src = URL.createObjectURL(file);
  document.getElementById("preview-source").style.display = "block";
  document.getElementById("text-source").style.display = "none";
});

document.getElementById("input-target").addEventListener("change", (e) => {
  const file = e.target.files[0];
  document.getElementById("preview-target").src = URL.createObjectURL(file);
  document.getElementById("preview-target").style.display = "block";
  document.getElementById("text-target").style.display = "none";
});

