![Image](https://github.com/user-attachments/assets/1a8abcf4-0433-463e-94f4-b56190c59a4d)
# PhishNet - Real-Time Phishing URL Detection System

PhishNet is a Flask-based web application that detects phishing URLs using machine learning. It uses a Random Forest classifier trained on a dataset of URLs labeled as safe or phishing.

## Features
- Real-time phishing URL detection
- Simple web interface
- Feature extraction from URLs (length, dots, entropy, HTTPS, IP presence)

## Dataset
The dataset (`urls_dataset.csv`) contains URLs and their labels:
- `1` = Phishing (fake)
- `0` = Safe (legitimate)

Example:
```
url,label
http://login-paypal-update-verify.com,1
https://www.google.com,0
```

## How to Run
1. **Install dependencies:**
   ```powershell
   pip install flask pandas scikit-learn tldextract requests whois
   ```
2. **Run the app:**
   ```powershell
   python main.py
   ```
3. **Open your browser:**
   Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## File Structure
- `main.py` — Main application and model code
- `urls_dataset.csv` — URL dataset
- `templates/index.html` — Web interface

## Improving Detection
- Add more phishing and safe URLs to `urls_dataset.csv` for better accuracy.
- Retrain the model by restarting the app after updating the dataset.

## License
MIT License
