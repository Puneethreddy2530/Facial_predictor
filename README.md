# AI Facial Insights PRO 🚀

A high-performance, privacy-focused facial attribute analysis application powered by a custom **Multi-Task EfficientNet** PyTorch model.

![App Preview](https://via.placeholder.com/800x400.png?text=Facial+Insights+PRO+Preview)

## ✨ Features
- **Real-time Inference**: Detects Age, Gender, and Race instantly.
- **Glassmorphism UI**: Premium, modern interface with drag-and-drop and camera support.
- **Privacy First**: All processing happens locally on your machine.
- **"Goated" Model**: Custom trained EfficientNet-B2 with Context Awareness.

## 🛠️ Tech Stack
- **Backend**: Python, FastAPI, PyTorch
- **Frontend**: HTML5, Vanilla JS, CSS3 (Glassmorphism)
- **AI/ML**: MTCNN (Face Detection), EfficientNet-B2 (Attribute Recognition)

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- CUDA compatible GPU (optional, for faster inference)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/facial-prediction-pro.git
   cd facial-prediction-pro
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

3. Download the Model:
   - Place your trained `best_model.pth` inside the `models/` directory.

### Running the App
1. **Start the Backend** (API):
   ```bash
   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
   ```

2. **Start the Frontend**:
   ```bash
   python -m http.server 3000 --directory frontend
   ```

3. Open your browser at `http://localhost:3000`.

## 📸 Camera Support
The app supports real-time webcam capture. Ensure you grant camera permissions when prompted.

---
*Built with ❤️ by AI + Human Collaboration*
