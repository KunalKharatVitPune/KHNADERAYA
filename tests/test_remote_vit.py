import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.models.vit_classifier import predict_dcrm_image

def test_remote_vit():
    # Use an existing image or create a dummy one
    image_path = os.path.join(os.path.dirname(__file__), '..', 'temp_vit_plot.png')
    
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found. Creating a dummy image.")
        from PIL import Image
        img = Image.new('RGB', (224, 224), color = 'red')
        img.save(image_path)

    print(f"Testing with image: {image_path}")
    
    # Call the function
    # Note: This will fail if the API is not reachable or if the code hasn't been updated yet.
    # We expect it to work AFTER we modify vit_classifier.py
    try:
        predicted_class, confidence, details = predict_dcrm_image(image_path)
        
        print("\n--- Prediction Results ---")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence}")
        print("\n--- Details ---")
        print(f"ViT Probs (Source should be remote): {details.get('vit_probs')}")
        print(f"Gemini Probs: {details.get('gemini_probs')}")
        print(f"Ensemble Scores: {details.get('ensemble_scores')}")
        
        if details.get('vit_probs'):
            print("\nSUCCESS: Received probabilities from ViT.")
        else:
            print("\nFAILURE: No ViT probabilities received.")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_remote_vit()
