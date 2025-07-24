// Thème clair/sombre
document.getElementById("theme-toggle").addEventListener("click", () => {
  document.body.classList.toggle("dark");
});

// Validation infos patient
document.getElementById("patient-form").addEventListener("submit", function (e) {
  e.preventDefault();
  const nom = document.getElementById("nom").value.trim();
  const prenom = document.getElementById("prenom").value.trim();
  const age = document.getElementById("age").value.trim();
  if (!nom || !prenom || !age) {
    document.getElementById("error-message").textContent = "Veuillez remplir tous les champs.";
  } else {
    document.getElementById("error-message").textContent = "";
    document.getElementById("upload").scrollIntoView({ behavior: "smooth" });
  }
});

// Validation radio
function validateRadioAndScroll() {
  const file = document.getElementById("radio-upload").files[0];
  if (!file) {
    document.getElementById("radio-error").textContent = "Veuillez importer une image.";
  } else {
    document.getElementById("radio-error").textContent = "";
    const reader = new FileReader();
    reader.onload = function (e) {
      document.getElementById("radio-preview").innerHTML = `<img src="${e.target.result}" alt="Radio">`;
    };
    reader.readAsDataURL(file);
    setTimeout(() => {
      document.getElementById("choix-modele").scrollIntoView({ behavior: "smooth" });
    }, 500);
  }
}

// Validation modèle + appel API
function validateModelAndScroll() {
  const model = document.getElementById("model-select").value;
  if (!model) {
    document.getElementById("model-error").textContent = "Veuillez sélectionner un modèle.";
  } else {
    document.getElementById("model-error").textContent = "";
    sendToAPI();  // Appel API ici
  }
}

// Bouton flottant vers chatbot
document.getElementById("chatbot-float").addEventListener("click", () => {
  document.getElementById("chatbot").scrollIntoView({ behavior: "smooth" });
});

// Chatbot IA enrichi par mots-clés
document.getElementById("chat-form").addEventListener("submit", function (e) {
  e.preventDefault();
  const input = document.getElementById("chat-input").value.toLowerCase();
  let response = "Je suis là pour vous aider avec vos questions dentaires.";

  if (input.includes("bonjour") || input.includes("salut")) {
    response = "Bonjour ! Je suis DentAI 🤖. Posez-moi une question sur votre santé dentaire.";
  } else if (input.includes("merci")) {
    response = "Avec plaisir ! Je suis toujours là si vous avez d'autres questions. 😊";
  } else if (input.includes("carie")) {
    response = "Une carie est une lésion de l’émail dentaire causée par des bactéries. Elle doit être soignée rapidement.";
  } else if (input.includes("abces") || input.includes("abcès")) {
    response = "Un abcès dentaire est une infection douloureuse, souvent accompagnée d’un gonflement. Consultez un dentiste rapidement.";
  } else if (input.includes("couronne")) {
    response = "Une couronne est une prothèse fixée sur une dent abîmée pour la protéger et lui redonner sa forme.";
  } else if (input.includes("douleur")) {
    response = "La douleur peut venir d’une carie, d’un abcès, ou même de la gencive. Une radio peut aider à identifier la cause.";
  } else if (input.includes("fonctionne") || input.includes("utiliser") || input.includes("mode d'emploi")) {
    response = "Ce site vous guide étape par étape : 1. Saisir vos infos, 2. Importer une radio, 3. Choisir un modèle IA, 4. Obtenir un diagnostic simulé.";
  } else if (input.includes("dentai")) {
    response = "DentAI est une plateforme d’intelligence artificielle dédiée au diagnostic dentaire à partir d’imagerie médicale.";
  } else if (input.includes("qui") && input.includes("peut")) {
    response = "Ce site peut être utilisé par les professionnels de santé, les étudiants ou toute personne curieuse d'explorer l'IA appliquée à la dentisterie.";
  } else {
    response = "Je n'ai pas compris votre question. Essayez avec des mots comme carie, douleur, abcès, etc.";
  }

  document.getElementById("chat-response").innerText = response;
});

// Fonction d'appel API (CNN ou YOLO)
async function sendToAPI() {
  const model = document.getElementById("model-select").value;
  const fileInput = document.getElementById("radio-upload");
  const file = fileInput.files[0];

  if (!file || !model) {
    alert("Merci de sélectionner un modèle et une image !");
    return;
  }

  const formData = new FormData();
formData.append("file", selectedFile); // selectedFile = le fichier image sélectionné

fetch(apiURL, {
  method: "POST",
  body: formData,
})
  .then((response) => {
    if (!response.ok) throw new Error("API error");
    return response.json();
  })
  .then((data) => {
    console.log("✅ Résultat API :", data);
    // Affiche le résultat dans ta page HTML ici
  })
  .catch((error) => {
    alert("Erreur lors de l'appel à l'API.");
    console.error(error);
  });

 const apiURL = model === "model1"
  ? "http://34.247.174.115:8000/predict"
  : "http://34.247.174.115:8000/detect";


  try {
    const response = await fetch(apiURL, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    let output = "";
    if (model === "model1") {
      output = `🦷 Résultat CNN : <strong>${data.prediction}</strong> (confiance : ${data.confidence})`;
    } else {
      output = "🦷 Résultats YOLO :<ul>" + data.detections.map(
        d => `<li>${d.label} - ${d.confidence}</li>`
      ).join('') + "</ul>";
      if (data.image_url) {
        output += `<div><img src="/images/predicted_image.jpg" alt="Image annotée YOLO" style="max-width:100%;margin-top:10px;border:2px solid #333;"></div>`;
      }
    }

    document.querySelector(".chat-box").innerHTML = output;
    document.getElementById("resultats").scrollIntoView({ behavior: "smooth" });

  } catch (error) {
    console.error("Erreur API :", error);
    alert("Erreur lors de l'appel à l'API.");
  }
}
