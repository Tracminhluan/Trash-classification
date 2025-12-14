import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ================================
# CONFIG
# ================================
MODEL_PATH = "model/keras_model.h5"
LABEL_PATH = "model/labels.txt"

model = load_model(MODEL_PATH)

labels = []
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(" ", 1)
        labels.append(parts[1] if len(parts) > 1 else line.strip())


# ================================
# APP CLASS
# ================================
class TrashClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TrashTerminator.net")
        self.root.geometry("900x650")
        self.root.configure(bg="#F4F6F9")

        self.image_data = None
        self.webcam_running = False

        self.build_ui()

    # ================================
    # UI Layout
    # ================================
    def build_ui(self):
        title = tk.Label(self.root, text="🗑 TrashTerminator.net",
                         font=("Segoe UI", 22, "bold"), fg="#333", bg="#F4F6F9")
        title.pack(pady=10)

        # ---------- BUTTON PANEL ----------
        btn_frame = tk.Frame(self.root, bg="#F4F6F9")
        btn_frame.pack(pady=10)

        self.make_button(btn_frame, "📁 Open Image", self.load_image).grid(row=0, column=0, padx=10)
        self.make_button(btn_frame, "📷 Start Webcam", self.start_webcam).grid(row=0, column=1, padx=10)
        self.make_button(btn_frame, "🛑 Stop Webcam", self.stop_webcam).grid(row=0, column=2, padx=10)

        # ---------- IMAGE PREVIEW ----------
        preview_frame = tk.LabelFrame(self.root, text=" Preview ", font=("Segoe UI", 12),
                                      bg="#E9ECEF", padx=10, pady=10)
        preview_frame.pack(pady=10)

        self.preview_label = tk.Label(preview_frame, text="No Image", bg="#E9ECEF")
        self.preview_label.pack()

        # ---------- PREDICTION OUTPUT ----------
        result_frame = tk.LabelFrame(self.root, text=" Prediction ", font=("Segoe UI", 12),
                                     bg="#E9ECEF", padx=10, pady=10)
        result_frame.pack(pady=10, fill="x")

        self.result_label = tk.Label(result_frame, text="", font=("Segoe UI", 16, "bold"), bg="#E9ECEF")
        self.result_label.pack()

        # ---------- PROBABILITIES ----------
        self.prob_frame = tk.LabelFrame(self.root, text=" Probabilities ", font=("Segoe UI", 12),
                                        bg="#E9ECEF", padx=10, pady=10)
        self.prob_frame.pack(pady=5, fill="x")

    # ================================
    # Nice Button
    # ================================
    def make_button(self, parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd, width=15,
                         font=("Segoe UI", 11), bg="#4A90E2", fg="white",
                         activebackground="#357ABD", activeforeground="white",
                         relief="solid", bd=1)

    # ================================
    # Open Image
    # ================================
    def load_image(self):
        if self.webcam_running:
            messagebox.showwarning("Warning", "Stop webcam first!")
            return

        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if not path:
            return

        img = Image.open(path)
        self.display_image(img)
        self.predict_from_image(img)

    # ================================
    # Start webcam
    # ================================
    def start_webcam(self):
        if self.webcam_running:
            return

        self.image_data = None
        self.preview_label.config(image="", text="Starting webcam...")
        self.clear_prediction()

        self.cap = cv2.VideoCapture(0)
        self.webcam_running = True
        self.update_webcam()

    # ================================
    # Stop webcam
    # ================================
    def stop_webcam(self):
        if not self.webcam_running:
            return

        self.webcam_running = False
        self.cap.release()

        self.preview_label.config(image="", text="Webcam stopped")
        self.clear_prediction()

    # ================================
    # Webcam update loop
    # ================================
    def update_webcam(self):
        if not self.webcam_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        display_img = img.resize((350, 350))

        imgtk = ImageTk.PhotoImage(display_img)
        self.preview_label.config(image=imgtk)
        self.preview_label.image = imgtk

        # Predict
        self.predict_from_image(img)

        self.root.after(50, self.update_webcam)  # ~20 FPS

    # ================================
    # Prediction
    # ================================
    def preprocess(self, img):
        img = img.resize((224, 224))
        arr = np.array(img).astype("float32")

        # CHUẨN HÓA THEO TEACHABLE MACHINE
        arr = (arr / 127.5) - 1.0

        return np.expand_dims(arr, axis=0)

    def predict_from_image(self, img):
        arr = self.preprocess(img)
        preds = model.predict(arr)[0]
        cls_index = int(np.argmax(preds))
        cls_name = labels[cls_index]

        self.result_label.config(text=f"Prediction: {cls_name}")

        for w in self.prob_frame.winfo_children():
            w.destroy()

        tk.Label(self.prob_frame, text="Probabilities:", font=("Arial", 12)).pack()

        for i, p in enumerate(preds):
            tk.Label(self.prob_frame, text=f"{labels[i]} — {p * 100:.2f}%").pack()

    # ================================
    # UI Helpers
    # ================================
    def clear_prediction(self):
        self.result_label.config(text="")
        for w in self.prob_frame.winfo_children():
            w.destroy()

    def display_image(self, img):
        img = img.resize((350, 350))
        imgtk = ImageTk.PhotoImage(img)
        self.preview_label.config(image=imgtk, text="")
        self.preview_label.image = imgtk
        self.image_data = img


# ================================
# RUN APP
# ================================
root = tk.Tk()
app = TrashClassifierApp(root)
root.mainloop()