import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tkinter import *
from PIL import Image, ImageTk
from datetime import datetime
import os

# ==== Cấu hình ====
MODEL_PATH = "../model/keras_model.h5"
LABEL_PATH = "../model/labels.txt"
SAVE_DIR = "../data/misclassified"
CONFIDENCE_THRESHOLD = 0.6  # Ngưỡng tự nhận 'nothing'

# ==== Tải model và labels ====
model = load_model(MODEL_PATH)
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip().split(maxsplit=1)[1] for line in f.readlines()]  # Bỏ số thứ tự nếu có

# ==== Hàm lưu ảnh sai ====
def save_misclassified(img, correct_label):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_folder = os.path.join(SAVE_DIR, correct_label)
    os.makedirs(label_folder, exist_ok=True)
    filename = os.path.join(label_folder, f"{now}.jpg")
    cv2.imwrite(filename, img)
    print(f"✅ Đã lưu ảnh sai vào: {filename}")

# ==== Ứng dụng chính ====
class TrashClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trash Classification")

        self.video = cv2.VideoCapture(0)
        self.panel = Label(root)
        self.panel.pack()

        self.label_text = StringVar()
        Label(root, textvariable=self.label_text, font=("Arial", 16)).pack()

        # Nút xác nhận
        Button(root, text="Đúng", command=self.mark_correct, width=10, bg="lightgreen").pack(side=LEFT, padx=10, pady=10)
        Button(root, text="Sai", command=self.mark_wrong, width=10, bg="lightcoral").pack(side=RIGHT, padx=10, pady=10)

        self.current_frame = None
        self.pred_label = ""
        self.update_frame()

    def update_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return

        self.current_frame = frame.copy()

        # === Tiền xử lý ảnh ===
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img_to_array(img)
        img = (img / 127.5) - 1.0  # ✅ Chuẩn hóa đúng theo Teachable Machine
        img = np.expand_dims(img, axis=0)

        # === Dự đoán ===
        preds = model.predict(img)[0]
        idx = np.argmax(preds)
        confidence = preds[idx]

        if confidence < CONFIDENCE_THRESHOLD:
            self.pred_label = "nothing"
        else:
            self.pred_label = labels[idx]

        self.label_text.set(f"{self.pred_label}: {confidence:.2f}")

        # === Hiển thị ảnh ===
        img = Image.fromarray(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB))
        img = ImageTk.PhotoImage(img)
        self.panel.configure(image=img)
        self.panel.image = img

        self.root.after(100, self.update_frame)

    def mark_correct(self):
        print("✅ Người dùng xác nhận: đúng.")

    def mark_wrong(self):
        print("❌ Người dùng xác nhận: sai.")
        self.ask_correct_label()

    def ask_correct_label(self):
        top = Toplevel(self.root)
        top.title("Chọn nhãn đúng")
        Label(top, text="Ảnh sai. Vui lòng chọn nhãn đúng:", font=("Arial", 12)).pack(pady=5)
        for label in labels:
            Button(top, text=label, width=20,
                   command=lambda l=label, w=top: self.save_and_close(l, w)).pack(pady=2)

    def save_and_close(self, correct_label, window):
        save_misclassified(self.current_frame, correct_label)
        window.destroy()

# ==== Khởi động ứng dụng ====
if __name__ == "__main__":
    root = Tk()
    app = TrashClassifierApp(root)
    root.mainloop()
