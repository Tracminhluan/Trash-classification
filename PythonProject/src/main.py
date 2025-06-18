import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tkinter import *
from PIL import Image, ImageTk
from datetime import datetime
import os
import tkinter.ttk as ttk
import time
import threading
import tkinter.font as font

# Đường dẫn đến model và nhãn
MODEL_PATH = "../model/keras_model.h5"
LABEL_PATH = "../model/labels.txt"
SAVE_DIR = "../data/misclassified"

# Load model và labels
model = load_model(MODEL_PATH)
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

def save_misclassified(img, correct_label):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_folder = os.path.join(SAVE_DIR, correct_label)
    os.makedirs(label_folder, exist_ok=True)
    filename = os.path.join(label_folder, f"{now}.jpg")
    cv2.imwrite(filename, img)
    print(f"Đã lưu ảnh sai vào: {filename}")

# Add at the top:
import tkinter.font as font

...

class TrashClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Trash Classification System")
        self.root.configure(bg="#befc95")

        self.video = cv2.VideoCapture(0)

        # Custom font
        self.default_font = font.Font(family="Helvetica", size=12)

        # Top Frame
        frame_top = Frame(root, bg="#4CAF50", pady=10)
        frame_top.pack(fill=X)

        Label(frame_top, text="🗑️ Smart Trash Classifier", font=("Helvetica", 20, "bold"), fg="white", bg="#4CAF50").pack()

        # Camera feed
        self.panel = Label(root, bg="#dfe6e9", bd=2, relief=SUNKEN)
        self.panel.pack(pady=10)

        # Prediction label
        self.label_text = StringVar()
        self.label_display = Label(root, textvariable=self.label_text, font=("Helvetica", 16, "bold"), 
                                   bg="white", fg="#2d3436", relief=SOLID, bd=2, padx=10, pady=5)
        self.label_display.pack(pady=8)

        # Confidence bar
        self.conf_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.conf_bar.pack(pady=5)
        create_tooltip(self.conf_bar, "📊 Confidence level of the current prediction")

        # Button frame
        btn_frame = Frame(root, bg="#f2f2f2")
        btn_frame.pack(pady=10)

        Button(btn_frame, text="✅ Đúng", command=self.mark_correct, width=12, bg="#81c784", font=self.default_font).pack(side=LEFT, padx=20)
        Button(btn_frame, text="❌ Sai", command=self.mark_wrong, width=12, bg="#e57373", font=self.default_font).pack(side=RIGHT, padx=20)

        self.current_frame = None
        self.pred_label = ""
        self.update_frame()

    def update_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return
        self.current_frame = frame.copy()

        img = cv2.resize(frame, (224, 224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        preds = model.predict(img)[0]
        idx = np.argmax(preds)
        self.pred_label = labels[idx]
        confidence = preds[idx]

        # Set label and color change based on confidence
        emoji_map = {
            "0 cans & bottles": "🥤",
            "1 general": "🗑️",
            "2 papers": "📄",
            "3 nothing": "❌"
        }

        label_icon = emoji_map.get(self.pred_label, "🔍")
        self.label_text.set(f"{label_icon} {self.pred_label.upper()}\n🎯 Confidence: {confidence:.2%}")

                
        self.conf_bar["value"] = confidence * 100

        # Convert image and show
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.panel.configure(image=img)
        self.panel.image = img

        self.root.after(100, self.update_frame)

    def mark_correct(self):
        print("✅ Xác nhận: đúng.")

    def mark_wrong(self):
        print("❌ Xác nhận: sai.")
        self.ask_correct_label()

    def ask_correct_label(self):
        top = Toplevel(self.root)
        top.title("Chọn nhãn đúng")
        top.configure(bg="#ffffff")

        Label(top, text="⚠️ Ảnh sai. Vui lòng chọn nhãn đúng:", font=("Arial", 12), bg="#ffffff").pack(pady=5)
        for i, label in enumerate(labels):
            Button(top, text=f"✔️ {label}", width=20,
                   command=lambda l=label, w=top: self.save_and_close(l, w), bg="#e0f7fa").pack(pady=2)

    def save_and_close(self, correct_label, window):
        save_misclassified(self.current_frame, correct_label)
        window.destroy()
    


def create_tooltip(widget, text):
    tooltip = Toplevel(widget)
    tooltip.withdraw()
    tooltip.overrideredirect(True)
    label = Label(tooltip, text=text, background="lightyellow", relief="solid", borderwidth=1)
    label.pack()
    
    def enter(event):
        x, y = event.x_root + 10, event.y_root + 10
        tooltip.geometry(f"+{x}+{y}")
        tooltip.deiconify()
    
    def leave(event):
        tooltip.withdraw()

    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)


    

if __name__ == "__main__":
    root = Tk()
    app = TrashClassifierApp(root)
    root.mainloop()

