# face_analyzer.py
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import cv2 # Using OpenCV for image handling

class FaceAnalyzer:
    """
    Handles face detection and embedding extraction using InsightFace.
    """
    def __init__(self, det_thresh=0.5, model_pack_name='buffalo_l', providers=None):
        """
        Initializes the FaceAnalysis app from InsightFace.

        Args:
            det_thresh (float): Detection confidence threshold.
            model_pack_name (str): Name of the model pack to use (e.g., 'buffalo_l', 'antelopev2').
                                   'buffalo_l' is generally recommended.
            providers (list, optional): List of ONNXRuntime providers.
                                        Defaults to ['CPUExecutionProvider'].
                                        Use ['CUDAExecutionProvider', 'CPUExecutionProvider'] for GPU.
        """
        if providers is None:
            providers = ['CPUExecutionProvider'] # Default to CPU

        print(f"Initializing InsightFace FaceAnalysis with model pack: {model_pack_name} and providers: {providers}")
        try:
            # Allowed modules specify which tasks to load models for
            self.app = FaceAnalysis(name=model_pack_name,
                                    allowed_modules=['detection', 'recognition'],
                                    providers=providers)
            self.app.prepare(ctx_id=0, det_thresh=det_thresh) # ctx_id=0 for CPU, >=0 for GPU
            print("InsightFace models loaded successfully.")
        except Exception as e:
            print(f"Error initializing InsightFace: {e}")
            print("Please ensure 'insightface' and 'onnxruntime' are installed correctly.")
            print("InsightFace models might need to be downloaded on first run, ensure internet connection.")
            # Depending on the error, ONNXRuntime GPU requirements might be missing.
            if 'CUDAExecutionProvider' in providers:
                print("If using GPU ('CUDAExecutionProvider'), ensure CUDA and cuDNN are installed and compatible with ONNXRuntime.")
            self.app = None

    def analyze_frame(self, frame: np.ndarray):
        """
        Detects faces in a frame and extracts their embeddings.

        Args:
            frame: The input image/frame (NumPy array in BGR format).

        Returns:
            A list of dictionaries. Each dictionary contains:
            'bbox': Bounding box coordinates [x1, y1, x2, y2].
            'kps': Keypoints (if available).
            'det_score': Detection confidence score.
            'embedding': The extracted face embedding (NumPy array).
            Returns an empty list if no faces are detected or if the model failed to initialize.
        """
        if self.app is None:
            print("FaceAnalyzer not initialized.")
            return []
        if frame is None or frame.size == 0:
            print("Received empty frame.")
            return []

        try:
            # InsightFace expects BGR format, which OpenCV usually provides
            faces = self.app.get(frame)

            # Prepare results in a more structured way, ensuring embedding is present
            results = []
            for face in faces:
                # Ensure the face object has the 'embedding' attribute
                if hasattr(face, 'embedding') and face.embedding is not None:
                    results.append({
                        'bbox': face.bbox.astype(int), # Bounding box [x1, y1, x2, y2]
                        'kps': face.kps,              # Keypoints
                        'det_score': face.det_score,  # Detection score
                        'embedding': face.normed_embedding # Use the normalized embedding for comparison
                    })
                else:
                    # This might happen if recognition is skipped or fails for a face
                     print(f"Warning: Face detected (score: {face.det_score:.2f}) but no embedding extracted. BBox: {face.bbox}")

            return results
        except Exception as e:
            print(f"Error during face analysis: {e}")
            return [] # Return empty list on error

    def get_single_embedding(self, frame: np.ndarray):
        """
        Analyzes a frame and returns the embedding of the most confident face.

        Args:
            frame: The input image/frame (NumPy array in BGR format).

        Returns:
            The embedding (NumPy array) of the most confident face, or None if no face is detected
            or if the model failed to initialize.
        """
        face_results = self.analyze_frame(frame)
        if not face_results:
            return None

        # Find the face with the highest detection score
        best_face = max(face_results, key=lambda x: x['det_score'])
        return best_face.get('embedding') # Use .get to avoid KeyError if embedding wasn't added

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    print("Testing FaceAnalyzer module...")
    # Initialize the analyzer (this might download models on first run)
    # Use providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] for GPU
    analyzer = FaceAnalyzer(providers=['CPUExecutionProvider'])

    if analyzer.app:
        # Create a dummy black image (replace with actual image loading or camera feed)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "Test Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        print("Analyzing dummy frame...")
        results = analyzer.analyze_frame(dummy_frame)

        if results:
            print(f"Found {len(results)} faces.")
            for i, face in enumerate(results):
                print(f"  Face {i+1}:")
                print(f"    BBox: {face['bbox']}")
                print(f"    Score: {face['det_score']:.4f}")
                print(f"    Embedding shape: {face['embedding'].shape}")
        else:
            print("No faces detected in the dummy frame.")

        # Example for getting single embedding
        # embedding = analyzer.get_single_embedding(dummy_frame)
        # if embedding is not None:
        #     print(f"\nExtracted single embedding with shape: {embedding.shape}")
        # else:
        #     print("\nCould not extract single embedding.")
    else:
        print("FaceAnalyzer failed to initialize.") 