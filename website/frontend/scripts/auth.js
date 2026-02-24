const API = "http://localhost:8000";

function getUser() {
  const u = localStorage.getItem("user");
  return u ? JSON.parse(u) : null;
}

function saveUser(user) {
  localStorage.setItem("user", JSON.stringify(user));
}

function logout() {
  localStorage.removeItem("user");
  window.location.href = "index.html";
}

// Met à jour la navbar selon l'état de connexion
document.addEventListener("DOMContentLoaded", () => {
  const user = getUser();
  const navLogin = document.getElementById("nav-login");
  const navDashboard = document.getElementById("nav-dashboard");
  const navLogout = document.getElementById("nav-logout");
  if (user) {
    if (navLogin) navLogin.style.display = "none";
    if (navDashboard) navDashboard.style.display = "inline";
    if (navLogout) navLogout.style.display = "inline";
  }
});
