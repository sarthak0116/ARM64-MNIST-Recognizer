import tkinter as tk
from tkinter import font
import numpy as np
import subprocess
import threading
from PIL import Image, ImageDraw, ImageFilter, ImageTk

CANVAS_SIZE  = 280
MODEL_SIZE   = 28
BRUSH_RADIUS = 16
BG_COLOR     = "#1a1a2e"
FG_COLOR     = "#ffffff"
ACCENT       = "#e94560"
BTN_HOVER    = "#0f3460"

class DigitDrawer:
    def __init__(self, root):
        self.root     = root
        self.root.title("Digit Recognizer")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(False, False)

        self.pil_image   = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw        = ImageDraw.Draw(self.pil_image)
        self.last_xy     = None

        self._predicting = False
        self._dirty      = False

        self._build_ui()
        self._poll()

    def _build_ui(self):
        title_font = font.Font(family="Helvetica", size=16, weight="bold")
        tk.Label(self.root, text="Digit Recognizer", font=title_font,
                 bg=BG_COLOR, fg=FG_COLOR).pack(pady=(16, 2))
        tk.Label(self.root, text="Draw",
                 bg=BG_COLOR, fg="#888888", font=("Helvetica", 10)).pack(pady=(0, 8))

        main_frame = tk.Frame(self.root, bg=BG_COLOR)
        main_frame.pack(padx=24)

        left = tk.Frame(main_frame, bg=BG_COLOR)
        left.pack(side="left")
        tk.Label(left, text="Drawing", bg=BG_COLOR,
                 fg="#888888", font=("Helvetica", 9)).pack()
        canvas_frame = tk.Frame(left, bg=ACCENT, padx=2, pady=2)
        canvas_frame.pack()
        self.canvas = tk.Canvas(canvas_frame,
                                width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>",       self._paint)
        self.canvas.bind("<ButtonPress-1>",   self._paint_start)
        self.canvas.bind("<ButtonRelease-1>", self._paint_end)

        right = tk.Frame(main_frame, bg=BG_COLOR)
        right.pack(side="left", padx=(16, 0), anchor="n", pady=(18, 0))
        tk.Label(right, text="Image sent to model",
                 bg=BG_COLOR, fg="#888888", font=("Helvetica", 9),
                 justify="center").pack()
        preview_frame = tk.Frame(right, bg=ACCENT, padx=2, pady=2)
        preview_frame.pack()
        self.preview_label = tk.Label(preview_frame, bg="black",
                                      width=140, height=140)
        self.preview_label.pack()

        self._make_btn(self.root, "Clear", self._clear, "#444455").pack(pady=12)

        result_font = font.Font(family="Helvetica", size=40, weight="bold")
        self.result_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.result_var,
                 font=result_font, bg=BG_COLOR, fg=ACCENT).pack()

        self.detail_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.detail_var,
                 bg=BG_COLOR, fg="#888888",
                 font=("Helvetica", 10)).pack(pady=(0, 16))

    def _make_btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd,
                         bg=color, fg=FG_COLOR, relief="flat",
                         font=("Helvetica", 12, "bold"),
                         padx=24, pady=8, cursor="hand2",
                         activebackground=BTN_HOVER, activeforeground=FG_COLOR,
                         borderwidth=0)

    def _paint_start(self, event):
        self.last_xy = (event.x, event.y)
        self._paint(event)

    def _paint_end(self, event):
        self.last_xy = None

    def _paint(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        self.canvas.create_oval(x-r, y-r, x+r, y+r,
                                fill="white", outline="white")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
        if self.last_xy:
            lx, ly = self.last_xy
            self.canvas.create_line(lx, ly, x, y, fill="white",
                                    width=r*2, capstyle=tk.ROUND,
                                    joinstyle=tk.ROUND)
            self.draw.line([lx, ly, x, y], fill=255, width=r*2)
        self.last_xy = (x, y)
        self._dirty = True
        self._update_preview()

    def _center_and_scale(self, img):
        arr  = np.array(img)
        rows = np.any(arr > 10, axis=1)
        cols = np.any(arr > 10, axis=0)
        if not rows.any():
            return Image.new("L", (MODEL_SIZE, MODEL_SIZE), 0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = img.crop((cmin, rmin, cmax+1, rmax+1))
        w, h    = cropped.size
        scale   = 20.0 / max(w, h)
        new_w   = max(1, int(w * scale))
        new_h   = max(1, int(h * scale))
        cropped = cropped.resize((new_w, new_h), Image.LANCZOS)
        canvas28 = Image.new("L", (MODEL_SIZE, MODEL_SIZE), 0)
        canvas28.paste(cropped, ((MODEL_SIZE - new_w)//2,
                                 (MODEL_SIZE - new_h)//2))
        return canvas28

    def _preprocess(self):
        blurred  = self.pil_image.filter(ImageFilter.GaussianBlur(radius=1))
        centered = self._center_and_scale(blurred)
        return np.array(centered).astype("float32") / 255.0

    def _update_preview(self):
        arr     = self._preprocess()
        display = Image.fromarray((arr * 255).astype(np.uint8)).resize(
                      (140, 140), Image.NEAREST)
        self._preview_img = ImageTk.PhotoImage(display)
        self.preview_label.config(image=self._preview_img)

    def _poll(self):
        if self._dirty and not self._predicting:
            self._dirty      = False
            self._predicting = True
            threading.Thread(target=self._run_inference, daemon=True).start()
        self.root.after(300, self._poll)

    def _run_inference(self):
        arr = self._preprocess()
        if arr.max() < 0.05:
            self.root.after(0, lambda: self.result_var.set(""))
            self.root.after(0, lambda: self.detail_var.set(""))
            self._predicting = False
            return

        arr.flatten().tofile("data/test_digit.bin")

        try:
            result = subprocess.run(["./test"], capture_output=True, text=True)
            output = result.stdout
        except FileNotFoundError:
            self.root.after(0, lambda: self.result_var.set("⚠️"))
            self.root.after(0, lambda: self.detail_var.set("'./test' not found — run 'make build' first"))
            self._predicting = False
            return

        predicted = None
        probs     = {}
        for line in output.splitlines():
            if "Predicted" in line and "True" not in line:
                try:
                    predicted = int(line.split(":")[1].strip())
                except ValueError:
                    pass
            if "digit" in line and ":" in line:
                try:
                    parts = line.strip().split()
                    d = int(parts[1].rstrip(":"))
                    p = float(parts[2])
                    probs[d] = p
                except (ValueError, IndexError):
                    pass

        if predicted is not None:
            top3   = sorted(probs.items(), key=lambda x: -x[1])[:3]
            detail = "     ".join(f"{d}: {p*100:.1f}%" for d, p in top3)
            self.root.after(0, lambda: self.result_var.set(f"→  {predicted}"))
            self.root.after(0, lambda: self.detail_var.set(detail))

        self._predicting = False

    def _clear(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw      = ImageDraw.Draw(self.pil_image)
        self.last_xy   = None
        self._dirty    = False
        self.result_var.set("")
        self.detail_var.set("")
        blank = ImageTk.PhotoImage(Image.new("L", (140, 140), 0))
        self._preview_img = blank
        self.preview_label.config(image=blank)


if __name__ == "__main__":
    root = tk.Tk()
    DigitDrawer(root)
    root.mainloop()