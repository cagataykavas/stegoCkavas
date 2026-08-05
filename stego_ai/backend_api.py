"""
stego_ai.backend_api
--------------------
Controller used by the optional desktop interface.
"""
import os
import json
import random
import subprocess
import sys
import shutil
from pathlib import Path
import cv2
import torch
import joblib 
import torch.nn.functional as F
import numpy as np

from stego_ai import stego_algorithms as sa
from stego_ai import pipeline
from stego_ai.train_stego_cnn import StegoNetLite 
from stego_ai.models import extract_features 

class StegoAPI:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.project_root, "models")
        self.cnn_path = os.path.join(self.models_dir, "stegocnn_best.pt")
        self.classic_path = os.path.join(self.models_dir, "models_binary.joblib")
        
        self.model_path = self.cnn_path if os.path.exists(self.cnn_path) else None
        self.classic_models = joblib.load(self.classic_path) if os.path.exists(self.classic_path) else {}

    # --- SINGLE PHOTO ---
    def single_embed(self, cover_path, save_dir, secret_message):
        results = {}
        os.makedirs(save_dir, exist_ok=True)
        stem = Path(cover_path).stem
        methods = ["lsb", "pvd", "dct", "dwt"]
        
        for alg in methods:
            out_name = f"{stem}_{alg}.png"
            key_name = f"{stem}_{alg}_key.json"
            out_path = os.path.join(save_dir, out_name)
            key_path = os.path.join(save_dir, key_name)
            seed = random.randint(1, 9999999)
            
            try:
                sa.process_image(cover_path, alg, "embed", secret_message, seed, out_path)
                key_data = {"algorithm": alg, "seed": seed, "info": "Load this to decrypt."}
                with open(key_path, 'w') as f: json.dump(key_data, f, indent=2)
                results[alg] = (out_path, key_path)
            except Exception as e:
                print(f"Embed error {alg}: {e}")
                results[alg] = None
        return results

    def single_extract(self, stego_path, key_path):
        try:
            with open(key_path, 'r') as f: key_data = json.load(f)
            return sa.process_image(stego_path, key_data["algorithm"], "extract", seed=key_data["seed"])
        except Exception as e: return f"Error: {e}"

    def get_residual_preview(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None: return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap = np.abs(lap); lap = (lap / (lap.max()+1e-8)) * 255.0
            p = img_path + "_res.png"
            cv2.imwrite(p, cv2.applyColorMap(lap.astype(np.uint8), cv2.COLORMAP_JET))
            return p
        except: return None

    # --- BATCH ---
    def generate_dataset_only(self, cover_dir, work_dir, bpp_val=0.4, progress_callback=None):
        if progress_callback: progress_callback(f"Generating Dataset (BPP={bpp_val})...")
        
        class PipelineArgs:
            cover_dir = None; work_dir = None; 
            bpp = bpp_val 
            algorithms = ["lsb", "pvd", "dct", "dwt"]
            seed = 42; normalize_covers = True
            force_generate = False; force_prepare = True
            train_ratio = 0.8; val_ratio = 0.1
            max_per_split = 400
            skip_generate = (bpp_val <= 0.01)
            skip_prepare = False; skip_multiclass = True
            save_models = True; feature_method = "residual_hist"
            feature_size = 64; dct_size = 8
            models = ["rf", "xgb", "svm"]
            pca_components = None

        args = PipelineArgs()
        args.cover_dir = cover_dir; args.work_dir = work_dir
        pipeline.run_pipeline(args)
        
        cls_p = os.path.join(work_dir, "saved_models", "models_binary.joblib")
        if os.path.exists(cls_p): self.classic_models = joblib.load(cls_p)
        if progress_callback: progress_callback("Done!")
        return os.path.join(work_dir, "classification_binary")

    def train_cnn_only(self, work_dir, epochs=25, progress_callback=None):
        ds_dir = os.path.join(work_dir, "classification_binary")
        if not os.path.exists(ds_dir): raise FileNotFoundError("Generate Data First!")
        if progress_callback: progress_callback(f"Training CNN ({epochs} epochs)...")
        
        out_mdl = os.path.join(work_dir, "cnn_model")
        cmd = [sys.executable, "-m", "stego_ai.train_stego_cnn", "--dataset-dir", ds_dir, 
               "--epochs", str(epochs), "--batch-size", "32", "--task", "binary", "--out-dir", out_mdl]
        subprocess.run(cmd, check=True)
        self.model_path = os.path.join(out_mdl, "stegocnn_best.pt")
        if progress_callback: progress_callback("CNN Ready!")
        return self.model_path

    # --- ANALYZE ---
    def get_classic_prediction(self, img_path, model_name):
        if not self.classic_models: return "Model not loaded.", "red"
        if model_name not in self.classic_models: return "Model not found.", "red"
        try:
            feat = extract_features(img_path, method="residual_hist").reshape(1, -1)
            clf = self.classic_models[model_name]
            pred = clf.predict(feat)[0]
            prob_str = ""
            if hasattr(clf, "predict_proba"):
                p = clf.predict_proba(feat)[0]
                conf = p[pred] * 100
                prob_str = f"({conf:.1f}%)"
            res = f"{model_name.upper()}: {'Stego' if pred==1 else 'Cover'} {prob_str}"
            return res, ("blue" if pred==1 else "green")
        except Exception as e: return f"Error: {e}", "red"

    def get_cnn_heatmap(self, img_path):
        m = self.model_path or self.cnn_path
        if not m or not os.path.exists(m): return None, "No CNN Model Found"
        try:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(m, map_location=dev)
            model = StegoNetLite(n_classes=2).to(dev)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            
            img = cv2.imread(img_path)
            g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (256,256))
            t = torch.from_numpy(g).float().unsqueeze(0).unsqueeze(0).to(dev)/255.0
            t.requires_grad_()
            
            logits = model(t)
            conf, pred = torch.max(F.softmax(logits, dim=1), 1)
            logits[0, pred].backward()
            
            sal = np.abs(t.grad.cpu().numpy()[0,0])
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            heat = cv2.applyColorMap(np.uint8(255*sal), cv2.COLORMAP_JET)
            over = cv2.addWeighted(cv2.resize(heat, (img.shape[1], img.shape[0])), 0.5, img, 0.5, 0)
            
            cv2.imwrite(img_path+"_heat.png", over)
            return img_path+"_heat.png", f"CNN: {['Cover','Stego'][pred]} ({conf.item()*100:.1f}%)"
        except Exception as e: return None, str(e)
