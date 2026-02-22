import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import sys

# Ensure backend can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stego_ai.backend_api import StegoAPI

class CodeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stego Lab 💣")
        self.root.geometry("1100x750")
        self.api = StegoAPI()
        
        style = ttk.Style()
        style.theme_use('clam')
        
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.t1 = ttk.Frame(self.nb); self.nb.add(self.t1, text="1. Hide Secret 🤫")
        self.t2 = ttk.Frame(self.nb); self.nb.add(self.t2, text="2. Reveal Secret 🔍")
        self.t3 = ttk.Frame(self.nb); self.nb.add(self.t3, text="3. AI Judge 🤖")
        
        self._setup_embed(); self._setup_extract(); self._setup_ai()
        self.status = tk.StringVar(value="Ready!"); ttk.Label(root, textvariable=self.status, relief=tk.SUNKEN).pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_embed(self):
        f = ttk.Frame(self.t1, padding=20); f.pack(fill='both', expand=True)
        top = ttk.LabelFrame(f, text="Config", padding=10); top.pack(fill='x')
        ttk.Button(top, text="Select Image", command=self.load_cover).pack(side=tk.LEFT)
        self.lbl_cover = ttk.Label(top, text="No file"); self.lbl_cover.pack(side=tk.LEFT)
        ttk.Label(top, text="Pass:").pack(side=tk.LEFT, padx=10)
        self.ent_pass = ttk.Entry(top, width=20); self.ent_pass.pack(side=tk.LEFT); self.ent_pass.insert(0, "Bomb!")
        ttk.Button(top, text="Fire!", command=self.run_hide).pack(side=tk.LEFT, padx=20)
        self.res_frame = ttk.Frame(f); self.res_frame.pack(fill='both', expand=True, pady=10)

    def load_cover(self):
        p = filedialog.askopenfilename(); 
        if p: self.cover_path = p; self.lbl_cover.config(text=os.path.basename(p))
    def run_hide(self):
        if hasattr(self, 'cover_path'): threading.Thread(target=self._hide_thread).start()
    def _hide_thread(self):
        res = self.api.single_embed(self.cover_path, os.path.join(os.path.dirname(self.cover_path), "stego_out"), self.ent_pass.get())
        self.root.after(0, lambda: self._show_res(res))
    def _show_res(self, res):
        for w in self.res_frame.winfo_children(): w.destroy()
        col=0
        for alg, val in res.items():
            if val:
                img, key = val
                fr = ttk.LabelFrame(self.res_frame, text=alg.upper()); fr.grid(row=0, column=col, padx=5)
                try: 
                    im = Image.open(img); im.thumbnail((150,150)); ph = ImageTk.PhotoImage(im)
                    l = ttk.Label(fr, image=ph); l.image=ph; l.pack()
                    ttk.Label(fr, text="Saved!").pack()
                except: pass
                col+=1

    def _setup_extract(self):
        f = ttk.Frame(self.t2, padding=20); f.pack(fill='both', expand=True)
        top = ttk.LabelFrame(f, text="Inputs", padding=10); top.pack(fill='x')
        ttk.Button(top, text="1. Stego Image", command=self.load_stego).pack(side=tk.LEFT)
        self.lbl_stego = ttk.Label(top, text="None"); self.lbl_stego.pack(side=tk.LEFT)
        ttk.Button(top, text="2. Key", command=self.load_key).pack(side=tk.LEFT, padx=15)
        self.lbl_key = ttk.Label(top, text="None"); self.lbl_key.pack(side=tk.LEFT)
        ttk.Button(top, text="Decrypt", command=self.run_decrypt).pack(side=tk.LEFT, padx=20)
        ttk.Button(top, text="X-Ray", command=self.run_xray).pack(side=tk.LEFT, padx=5)
        self.lbl_msg = ttk.Label(f, text="...", font=("Arial", 16), background="#eee"); self.lbl_msg.pack(fill='x', pady=10)
        self.vis_frame = ttk.LabelFrame(f, text="Visual Inspection", padding=10); self.vis_frame.pack(fill='both', expand=True)
        self.lbl_prev = ttk.Label(self.vis_frame); self.lbl_prev.pack(expand=True)

    def load_stego(self): 
        p = filedialog.askopenfilename(); 
        if p: self.stego_path=p; self.lbl_stego.config(text=os.path.basename(p)); self._show_prev(p)
    def load_key(self): 
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")]); 
        if p: self.key_path=p; self.lbl_key.config(text=os.path.basename(p))
    def _show_prev(self, p):
        try: im = Image.open(p); im.thumbnail((350,350)); ph = ImageTk.PhotoImage(im); self.lbl_prev.config(image=ph); self.lbl_prev.image=ph
        except: pass
    def run_decrypt(self):
        if hasattr(self, 'stego_path') and hasattr(self, 'key_path'):
            msg = self.api.single_extract(self.stego_path, self.key_path)
            color = "red" if "Error" in msg else "blue"
            self.lbl_msg.config(text=msg, foreground=color)
    def run_xray(self):
        if hasattr(self, 'stego_path'):
            p = self.api.get_residual_preview(self.stego_path)
            if p: self._show_prev(p)

    def _setup_ai(self):
        f = ttk.Frame(self.t3, padding=20); f.pack(fill='both', expand=True)
        left = ttk.LabelFrame(f, text="Training Lab", padding=10); left.pack(side=tk.LEFT, fill='both', expand=True)
        ttk.Button(left, text="Select Cover Folder", command=self.sel_data).pack(fill='x')
        self.lbl_data = ttk.Label(left, text="No folder selected"); self.lbl_data.pack()
        f_bpp = ttk.Frame(left); f_bpp.pack(fill='x', pady=5)
        ttk.Label(f_bpp, text="Noise (BPP) [0=Skip Gen]:").pack(side=tk.LEFT)
        self.scale_bpp = tk.Scale(f_bpp, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL)
        self.scale_bpp.set(0.1); self.scale_bpp.pack(side=tk.LEFT, fill='x', expand=True)
        ttk.Button(left, text="1. Generate Dataset", command=self.run_gen).pack(fill='x', pady=5)
        f_ep = ttk.Frame(left); f_ep.pack(fill='x')
        ttk.Label(f_ep, text="Epochs:").pack(side=tk.LEFT)
        self.ent_ep = ttk.Entry(f_ep, width=5); self.ent_ep.pack(side=tk.LEFT); self.ent_ep.insert(0,"25")
        ttk.Button(left, text="2. Train CNN", command=self.run_train).pack(fill='x', pady=5)
        self.log_txt = tk.Text(left, height=10, width=30); self.log_txt.pack(fill='both', expand=True)
        
        right = ttk.LabelFrame(f, text="Verification", padding=10); right.pack(side=tk.RIGHT, fill='both', expand=True)
        ttk.Button(right, text="Test Image", command=self.load_ai_test).pack()
        f_mod = ttk.Frame(right); f_mod.pack(fill='x', pady=5)
        ttk.Label(f_mod, text="Judge:").pack(side=tk.LEFT)
        self.combo = ttk.Combobox(f_mod, values=["CNN (Heatmap)", "Random Forest", "XGBoost", "SVM"], state="readonly")
        self.combo.current(0); self.combo.pack(side=tk.LEFT)
        ttk.Button(right, text="Analyze", command=self.run_analyze).pack(pady=5)
        self.lbl_res = ttk.Label(right, text="???", font=("Arial", 14)); self.lbl_res.pack()
        self.lbl_heat = ttk.Label(right); self.lbl_heat.pack(expand=True)

    def sel_data(self): 
        d = filedialog.askdirectory(); 
        if d: self.data_dir=d; self.lbl_data.config(text=os.path.basename(d))
    def log(self, m): self.log_txt.insert(tk.END, m+"\n"); self.log_txt.see(tk.END)
    def run_gen(self):
        if not hasattr(self, 'data_dir'): messagebox.showerror("Error", "Select Folder!"); return
        bpp = self.scale_bpp.get()
        def t():
            try: 
                wd = os.path.join(self.data_dir, "ai_workspace")
                self.api.generate_dataset_only(self.data_dir, wd, bpp, self.log)
            except Exception as e: self.log(f"Err: {e}")
        threading.Thread(target=t).start()
    def run_train(self):
        if not hasattr(self, 'data_dir'): return
        try: ep = int(self.ent_ep.get())
        except: ep = 25
        def t():
            try:
                wd = os.path.join(self.data_dir, "ai_workspace")
                self.api.train_cnn_only(wd, ep, self.log)
            except Exception as e: self.log(f"Err: {e}")
        threading.Thread(target=t).start()
    def load_ai_test(self):
        p = filedialog.askopenfilename(); 
        if p: self.ai_path=p; im=Image.open(p); im.thumbnail((300,300)); ph=ImageTk.PhotoImage(im); self.lbl_heat.config(image=ph); self.lbl_heat.image=ph
    def run_analyze(self):
        if not hasattr(self, 'ai_path'): return
        mod = self.combo.get()
        if "CNN" in mod:
            hp, txt = self.api.get_cnn_heatmap(self.ai_path)
            if hp:
                im=Image.open(hp); im.thumbnail((300,300)); ph=ImageTk.PhotoImage(im); self.lbl_heat.config(image=ph); self.lbl_heat.image=ph
            self.lbl_res.config(text=txt)
        else:
            txt, col = self.api.get_classic_prediction(self.ai_path, mod)
            self.lbl_res.config(text=txt, foreground=col)
            im=Image.open(self.ai_path); im.thumbnail((300,300)); ph=ImageTk.PhotoImage(im); self.lbl_heat.config(image=ph); self.lbl_heat.image=ph

if __name__ == "__main__":
    root = tk.Tk()
    app = CodeGUI(root)
    root.mainloop()