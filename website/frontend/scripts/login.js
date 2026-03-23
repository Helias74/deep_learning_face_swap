

document.getElementById("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;
  const errorEl = document.getElementById("error");

  try {
    const res = await fetch(`${API}/users/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (!res.ok) {
      const data = await res.json();
      errorEl.textContent = data.detail || "Erreur de connexion";
      errorEl.style.display = "block";
      return;
    }

    const data = await res.json();
    saveUser(data.user);
    window.location.href = "swap.html";

  } catch (err) {
    errorEl.textContent = "Impossible de contacter le serveur";
    errorEl.style.display = "block";
  }
});
