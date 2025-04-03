# Face Authentication System

This project implements a real-time face authentication system using OpenCV for camera access, InsightFace for face detection and recognition (embedding extraction), and PostgreSQL with the pgvector extension for storing and comparing face embeddings.

## Features

*   **Real-time Face Detection:** Detects faces in the webcam feed.
*   **Face Recognition:** Generates facial embeddings using state-of-the-art deep learning models (via InsightFace).
*   **Database Storage:** Stores known face embeddings along with names in a PostgreSQL database.
*   **Vector Similarity Search:** Uses `pgvector` for efficient cosine similarity search to find matching faces.
*   **Authentication:** Identifies known individuals in the video stream and prints an "authenticated" message.
*   **Registration Tool:** Includes a command-line script to easily register new faces into the system.

## Project Structure

```
.
├── database.py         # Handles PostgreSQL connection and queries (using pgvector)
├── face_analyzer.py    # Encapsulates InsightFace model loading and analysis
├── main_auth.py        # Main script for running the authentication loop
├── README.md           # This file
├── register_face.py    # Script to register new faces via webcam
└── requirements.txt    # Python package dependencies
```

## Setup

1.  **Prerequisites:**
    *   Python 3.8+
    *   PostgreSQL server (e.g., v13+)
    *   Git (for cloning if necessary)
    *   C++ compiler and CMake (required by `dlib` which might be installed as a dependency of other packages, although not directly used here, it's good practice for face-related projects)

2.  **Clone the Repository (Optional):**
    ```bash
    # git clone <repository-url>
    # cd <repository-directory>
    ```

3.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `insightface` will download models (`buffalo_l` by default) the first time `FaceAnalyzer` is initialized. Ensure you have an internet connection.*
    
    **Execution Acceleration:**
    *   **macOS (Automatic):** The scripts automatically detect macOS (`Darwin`) and prioritize the `CoreMLExecutionProvider` for ONNX Runtime. This allows leveraging Apple's Core ML framework, which may utilize Metal Performance Shaders (MPS) on the GPU or the Neural Engine (ANE) for significantly faster model inference on Apple Silicon and compatible Intel Macs.
    *   **NVIDIA GPU (Manual):** If you have a compatible NVIDIA GPU, CUDA, and cuDNN installed on Linux/Windows, you can install the GPU version of ONNX Runtime:
        ```bash
        # Find the correct version at https://onnxruntime.ai/
        # Example: pip install onnxruntime-gpu
        ```
        *The scripts currently default to CPU on non-macOS systems. You would need to modify the `INSIGHTFACE_PROVIDERS` logic in `register_face.py` and `main_auth.py` to include `'CUDAExecutionProvider'` if you install the GPU package.*
    *   **CPU (Default):** If neither Core ML nor CUDA is used, inference will run on the CPU via the `CPUExecutionProvider`.

5.  **Set up PostgreSQL Database:**
    *   Ensure your PostgreSQL server is running.
    *   Connect to PostgreSQL (e.g., using `psql`).
    *   Create a database user and password (if needed).
    *   Create the database:
        ```sql
        CREATE DATABASE face_auth_db;
        ```
    *   Connect to the new database:
        ```sql
        \c face_auth_db
        ```
    *   Enable the `pgvector` extension:
        ```sql
        CREATE EXTENSION IF NOT EXISTS vector;
        ```
    *   Grant privileges to your user if necessary.

6.  **Configure Database Connection:**
    *   Edit `database.py` and update the following variables with your actual PostgreSQL credentials:
        ```python
        DB_NAME = "face_auth_db"
        DB_USER = "your_db_user"
        DB_PASSWORD = "your_db_password"
        DB_HOST = "localhost" # Or your DB host
        DB_PORT = "5432"      # Or your DB port
        ```
    *   Alternatively, set these as environment variables (`PG_DB`, `PG_USER`, `PG_PASSWORD`, `PG_HOST`, `PG_PORT`).

## Usage

1.  **Register Faces:**
    *   Run the registration script:
        ```bash
        python register_face.py
        ```
    *   A window showing your webcam feed will appear.
    *   Position your face clearly in the frame (a green box should appear if detected).
    *   Press the 'c' key to capture the image.
    *   If successful, you'll be prompted in the terminal to enter a name for the person. Type the name and press Enter.
    *   Repeat for each person you want to register.
    *   Press 'q' in the window to quit the registration script.

2.  **Run Authentication:**
    *   Run the main authentication script:
        ```bash
        python main_auth.py
        ```
    *   A window will show the webcam feed.
    *   Detected faces will have bounding boxes drawn around them.
        *   **Green Box:** Recognized face (name and distance score shown). An "AUTHENTICATED: [Name]" message will be printed to the console once per frame per recognized person.
        *   **Red Box:** Unknown face.
    *   Press 'q' in the window to quit the authentication script.

## Customization

*   **Execution Providers:** The scripts automatically attempt to use Core ML (`CoreMLExecutionProvider`) on macOS for hardware acceleration. On other operating systems, they default to CPU (`CPUExecutionProvider`). You can modify the logic near the top of `register_face.py` and `main_auth.py` if you want to manually force specific providers (e.g., add CUDA support).
*   **Distance Threshold:** Adjust `DISTANCE_THRESHOLD` in `database.py` (default is 0.5). Lower values make recognition stricter (faces must be more similar). Experiment to find a good value for your use case and the `insightface` model. Cosine distance typically ranges from 0 (identical) to 2.
*   **Recognition Interval:** Modify `RECOGNITION_INTERVAL_SECONDS` in `main_auth.py` to change how often full recognition (including database lookup) is performed. Lower values increase CPU/GPU usage but provide more real-time updates.
*   **Camera Index:** Change `CAMERA_INDEX` in `register_face.py` and `main_auth.py` if your desired webcam is not the default (index 0).
*   **InsightFace Model:** You can change the `model_pack_name` in `face_analyzer.py` (e.g., to `'antelopev2'`) if needed, but ensure the `VECTOR(512)` dimension in `database.py` matches the model's output dimension.
*   **Database Index:** The `database.py` script includes commented-out code to create an `hnsw` index (`idx_face_embeddings_hnsw`) on the `embedding` column. Uncommenting and running `initialize_database` (e.g., by running `python database.py` once) can significantly speed up searches on large datasets. 