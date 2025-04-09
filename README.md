
---

```markdown
# 🎙️ Interview Test Platform

This is an interactive Streamlit-based web app where candidates can upload their CVs, answer audio-based interview questions, and receive scores based on their spoken responses. The app evaluates candidate performance and stores results in a PostgreSQL database.

---

## 🚀 Features

- 📄 Resume Upload & Name Extraction
- ❓ AI-generated Interview Questions
- 🎤 Audio Recording of Responses
- 🧠 Speech-to-Text Transcription
- 🧾 Automatic Scoring System
- 📊 Score Summary (Total, Percentage, Average per Question)
- 💾 Candidate Data + Marks Saved to PostgreSQL

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Database**: PostgreSQL
- **Audio**: Pydub, audiorecorder
- **Speech-to-Text**: OpenAI Whisper / Custom Transcriber

---

## 📦 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/rizvanasherf/skill-test
   cd skill-test
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up `.env` File**
   Create a `.env` file in the root directory with:

   ```env
   DB_HOST="localhost"
   DB_PORT="5432"
   DB_NAME="test_platform"
   DB_USER="postgres"
   DB_PASSWORD="your_password"
   ```

5. **Start PostgreSQL**
   Make sure your PostgreSQL server is running and the database `test_platform` is created:

   ```sql
   CREATE DATABASE test_platform;
   ```

6. **Run the App**
   ```bash
   streamlit run app.py
   ```

---

## 🧪 Sample Output

- 👤 Welcome Page with Candidate Name
- 🎤 Live Audio Recording for Each Question
- 🧠 AI-based Scoring of Transcript
- 📊 Final Score Summary: Total, Percentage, Average
- ✅ Data stored in PostgreSQL

---

## 📁 Folder Structure

```
├── app.py                 # Main Streamlit app
├── utils.py               # Helper functions (DB, scoring, etc.)
├── .env                   # Database credentials
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── assets/                # Audio, fonts, styles (optional)
```

---

## 🙌 Credits

- [Streamlit](https://streamlit.io/)
- [Pydub](https://github.com/jiaaro/pydub)
- [PostgreSQL](https://www.postgresql.org/)
- [OpenAI Whisper](https://github.com/openai/whisper) (or your own STT engine)

---

## 📬 Contact

For support or collaboration, contact: [rizvanwork293@gmail.com.com](mailto:rizvanwork293@example.com)

---

```

