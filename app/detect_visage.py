import cv2

# Charger le classificateur de visages
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces(image_path, output_path="result.jpg"):
    # Lire l'image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image introuvable")

    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détection des visages
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Dessiner les rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),  # vert
            2
        )

    # Sauvegarde
    cv2.imwrite(output_path, img)
    print(f"{len(faces)} visage(s) détecté(s)")
    print(f"Image sauvegardée : {output_path}")

if __name__ == "__main__":
    detect_faces("photo.jpg")
