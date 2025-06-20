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
import pygame
from screeninfo import get_monitors

# Đường dẫn đến model và nhãn
MODEL_PATH = "../model/keras_model.h5"
LABEL_PATH = "../model/labels.txt"
SAVE_DIR = "../data/misclassified" # This directory is for misclassified items

# --- NEW FEATURE: SAVE CORRECTLY CLASSIFIED ---
SAVE_CORRECT_DIR = "../data/correctly_classified" # New directory for correctly classified items
# --- END NEW FEATURE ---

# Load model và labels
model = load_model(MODEL_PATH)
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize pygame mixer
pygame.mixer.init()

def save_misclassified(img, correct_label):
    """Saves an image to the misclassified directory under the correct label folder."""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_folder = os.path.join(SAVE_DIR, correct_label)
    os.makedirs(label_folder, exist_ok=True)
    filename = os.path.join(label_folder, f"{now}.jpg")
    cv2.imwrite(filename, img)
    print(f"Đã lưu ảnh sai vào: {filename}") # (Saved misclassified image to:)

# --- NEW FEATURE: SAVE CORRECTLY CLASSIFIED ---
def save_correctly_classified(img, predicted_label):
    """Saves an image to the correctly classified directory under its predicted label folder."""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_folder = os.path.join(SAVE_CORRECT_DIR, predicted_label)
    os.makedirs(label_folder, exist_ok=True)
    filename = os.path.join(label_folder, f"{now}.jpg")
    cv2.imwrite(filename, img)
    print(f"Đã lưu ảnh đúng vào: {filename}") # (Saved correctly classified image to:)
# --- END NEW FEATURE ---

# Function to play sound
def play_sound(file_path, repeats=0): # repeats=0 means play once, 1 means play twice, etc.
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play(loops=repeats) # Play the sound 'repeats' times (0 for once, 1 for twice, 2 for three times)
    except pygame.error as e:
        print(f"Error playing sound {file_path}: {e}")


class TrashClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Trash Classification System")
        self.root.configure(bg="#befc95")

        # --- FIX: Variable Name Typo in Scaling Calculation ---
        # Added feature ------------------------ Scaling Screen --------------------------------------
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()



        
        print("Primary screen width:", screen_width)
        print("Primary screen height:", screen_height)
        self.scale = min(screen_width / 1920, screen_height / 1080) # Corrected variable names here
        # Added debug prints for scale
        print(f"DEBUG: Initial calculated self.scale: {self.scale}")
        # --------------------------------------------------------------------------------------------
        # --- END FIX ---

        self.video = cv2.VideoCapture(0)

        # Custom font
        # --- FIX: SCALING FOR FONT SIZES ---
        # Ensure minimum font sizes
        min_font_size_default = 8
        min_font_size_large = 12
        min_font_size_medium = 10

        self.default_font = font.Font(family="Helvetica", size=max(min_font_size_default, int(12 * self.scale)))
        self.large_font_size = max(min_font_size_large, int(20 * self.scale))
        self.medium_font_size = max(min_font_size_medium, int(16 * self.scale))
        # --- END FIX ---

        # Top Frame
        # --- FIX: SCALING FOR PADDING ---
        frame_top = Frame(root, bg="#4CAF50", pady=max(1, int(10 * self.scale))) # Ensure minimum padding
        # --- END FIX ---
        frame_top.pack(fill=X)

        Label(frame_top, text="🗑️ Smart Trash Classifier", font=("Helvetica", self.large_font_size, "bold"), fg="white", bg="#4CAF50").pack()

        # Camera feed
        # --- FIX: SCALING FOR PANEL DIMENSIONS ---
        self.panel_width = max(100, int(640 * self.scale)) # Ensure minimum width
        self.panel_height = max(75, int(480 * self.scale))  # Ensure minimum height

        self.panel = Label(root, width=self.panel_width, height=self.panel_height, bg="#dfe6e9", bd=max(1, int(2*self.scale)), relief=SUNKEN) # Ensure minimum border
        self.panel.pack(pady=max(1, int(10 * self.scale))) # Ensure minimum padding
        # --- END FIX ---


        # Prediction label
        self.label_text = StringVar()
        # --- FIX: SCALING FOR PADDING AND BORDER ---
        self.label_display = Label(root, textvariable=self.label_text, font=("Helvetica", self.medium_font_size, "bold"), 
                                     bg="white", fg="#2d3436", relief=SOLID, bd=max(1, int(2*self.scale)), padx=max(1, int(10*self.scale)), pady=max(1, int(5*self.scale))) # Ensure minimums
        self.label_display.pack(pady=max(1, int(8 * self.scale))) # Ensure minimum padding
        # --- END FIX ---

        # Confidence bar
        # --- FIX: SCALING FOR BAR LENGTH AND PADDING ---
        bar_length = max(50, int(300 * self.scale)) # Ensure minimum length
        self.conf_bar = ttk.Progressbar(root, orient="horizontal", length=bar_length, mode="determinate")
        self.conf_bar.pack(pady=max(1, int(5*self.scale))) # Ensure minimum padding
        # --- END FIX ---
        create_tooltip(self.conf_bar, "📊 Confidence level of the current prediction")

        # Button frame
        # --- FIX: SCALING FOR PADDING ---
        btn_frame = Frame(root, bg="#f2f2f2")
        btn_frame.pack(pady=max(1, int(10 * self.scale))) # Ensure minimum padding

        # --- FIX: SCALING FOR BUTTONS (width and padding) ---
        min_button_width_chars = 5 # Minimum width in character units
        min_button_padx = 5

        Button(btn_frame, text="✅ Đúng", command=self.mark_correct, width=max(min_button_width_chars, int(12 * self.scale)), bg="#81c784", font=self.default_font).pack(side=LEFT, padx=max(min_button_padx, int(20 * self.scale)))
        Button(btn_frame, text="❌ Sai", command=self.mark_wrong, width=max(min_button_width_chars, int(12 * self.scale)), bg="#e57373", font=self.default_font).pack(side=RIGHT, padx=max(min_button_padx, int(20 * self.scale)))
        # --- END FIX ---

        self.current_frame = None
        self.pred_label = ""
        self.update_frame()

    def update_frame(self):
        ret, frame = self.video.read()
        if not ret:
            # --- CAMERA ERROR HANDLING ---
            # Handle the case where the camera is not available or disconnected
            self.label_text.set("❌ Camera Error: No feed available. Please check connection.")
            self.panel.configure(image='') # Clear previous image
            self.panel.image = None # Release reference
            self.root.after(1000, self.update_frame) # Try again after a short delay
            return
            # --- END CAMERA ERROR HANDLING ---

        self.current_frame = frame.copy()

        # Resize for model prediction
        img_for_pred = cv2.resize(frame, (224, 224))
        img_for_pred = img_to_array(img_for_pred) / 255.0
        img_for_pred = np.expand_dims(img_for_pred, axis=0)
        
        preds = model.predict(img_for_pred)[0] # Use img_for_pred for model prediction
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

        # Convert original frame for display
        img_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_display = Image.fromarray(img_display)
        
        # --- FIX: SCALING FOR IMAGE DISPLAY ---
        # Resize for display on panel, maintaining aspect ratio if needed, or just fit to panel size
        img_display = img_display.resize((self.panel_width, self.panel_height), Image.Resampling.LANCZOS)
        # --- END FIX ---
        
        img_display_tk = ImageTk.PhotoImage(img_display)
        self.panel.configure(image=img_display_tk)
        self.panel.image = img_display_tk

        self.root.after(100, self.update_frame)

    def mark_correct(self):
        print("✅ Xác nhận: đúng.")
        # Play sound
        if self.pred_label == "0 cans & bottles":
            play_sound("binRed.mp3", repeats=2)
        elif self.pred_label == "1 general":
            play_sound("binYellow.mp3", repeats=2)
        elif self.pred_label == "2 papers":
            play_sound("binBlue.mp3", repeats=2)
        # No sound for "3 nothing" as per requirements

        # --- NEW FEATURE: SAVE CORRECTLY CLASSIFIED ---
        # Save the correctly classified image
        if self.current_frame is not None:
            save_correctly_classified(self.current_frame, self.pred_label)
        # --- END NEW FEATURE ---


    def mark_wrong(self):
        print("❌ Xác nhận: sai.")
        self.ask_correct_label()

    def ask_correct_label(self):
        top = Toplevel(self.root)
        top.title("Chọn nhãn đúng") # (Choose correct label)
        top.configure(bg="#ffffff")

        # --- FIX: SCALING FOR TOPLEVEL BUTTONS AND LABEL ---
        # Ensure minimum font sizes and widths for visibility
        min_label_font_size = 8
        min_button_width_chars = 10 # Minimum width in character units
        min_button_pady = 2
        min_label_pady = 2
        
        label_font_size_scaled = max(min_label_font_size, int(12 * self.scale))

        Label(top, text="⚠️ Ảnh sai. Vui lòng chọn nhãn đúng:", 
              font=("Arial", label_font_size_scaled), bg="#ffffff").pack(pady=max(min_label_pady, int(5*self.scale)))
        
        for i, label_text in enumerate(labels): # Renamed 'label' to 'label_text' to avoid confusion with the Label widget
            Button(top, text=f"✔️ {label_text}", 
                   width=max(min_button_width_chars, int(20 * self.scale)),
                   command=lambda l=label_text, w=top: self.save_and_close(l, w), 
                   bg="#e0f7fa", 
                   font=self.default_font).pack(pady=max(min_button_pady, int(2*self.scale)))
        # --- END FIX ---

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

    # Quit pygame mixer when the application closes
    pygame.mixer.quit()