def get_cnn_heatmap(self, img_path):
        """
        Loads the trained model, predicts the image, and generates a saliency map.
        Returns: (heatmap_overlay_image, result_text)
        """
        if not self.model_path or not os.path.exists(self.model_path):
            return None, "No Model Found. Please Train First!"

        try:
            # 1. Load Model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint = torch.load(self.model_path, map_location=device)
            
            # Re-init model structure (Must match training!)
            # We assume binary for now as per your requirement
            model = StegoNetLite(n_classes=2).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()

            # 2. Preprocess Image (Grayscale -> Resize -> Tensor)
            # This must match the training preprocessing
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: return None, "Read Error"
            
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # Resize to 256x256 as per default training patch size
            gray_resized = cv2.resize(gray, (256, 256)) 
            
            # Prepare Tensor (1, 1, 256, 256)
            input_tensor = torch.from_numpy(gray_resized).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(device)
            input_tensor.requires_grad_() # Enable gradients for heatmap

            # 3. Predict
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            confidence, pred_class = torch.max(probs, 1)
            
            # 4. Generate Saliency (Input Gradients)
            # We want to see what pixels triggered this specific class prediction
            score = logits[0, pred_class]
            model.zero_grad()
            score.backward()
            
            gradients = input_tensor.grad.data.cpu().numpy()[0,0]
            saliency = np.abs(gradients)
            # Normalize to 0..1
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            
            # 5. Create Visualization Overlay
            heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0])) # Resize back to original
            overlay = cv2.addWeighted(heatmap, 0.5, img_bgr, 0.5, 0)
            
            # Save temp file for GUI to display
            out_path = img_path + "_heatmap.png"
            cv2.imwrite(out_path, overlay)
            
            classes = ["Cover", "Stego"] # Assuming binary order
            result_text = f"Prediction: {classes[pred_class]} ({confidence.item()*100:.1f}%)"
            
            return out_path, result_text

        except Exception as e:
            print(f"Heatmap Error: {e}")
            return None, f"Error: {str(e)}"