import io
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

import streamlit as st
from audiorecorder import audiorecorder

from utils import (
    calculate_score,
    extract_resume_data,
    save_audio_to_wav,
    generate_llm_questions,
    score_check,
    text_to_speech,
    transcribe_audio,
    uniquequestion,
    save_candidate_results
)


# Set page configuration
st.set_page_config(
    page_title="DataSkillTest.AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_dotenv()

# Apply custom CSS for black background and silver/gray text
with open("asset/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state variables
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'

if 'uploaded_cv' not in st.session_state:
    st.session_state['uploaded_cv'] = None


# ----------------- PAGE FUNCTIONS -----------------
def home_page():
    """
        Render the home page of the DataSkillAI platform.

        This function displays the introductory content and overview of the Data Scientist Skill 
        Assessment Platform. It includes:
        
        - A welcoming header and description of the platform's purpose.
        - A detailed explanation of how the platform works, including resume upload, skill assessment, 
        personalized feedback, and recruiter visibility.
        - A list of key skills that are assessed (e.g., programming, statistics, ML, SQL, etc.).
        - A 'Start Assessment' button that navigates the user to the resume upload page when clicked.
        
        This function uses Streamlit to layout the content in columns and apply custom HTML/CSS 
        styling for enhanced presentation.
    """
    st.markdown("<h1 class='main-title'>DataSkillAI</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div >
            <p class="silver-text" style="font-size: 20px; text-align: center;">
                Welcome to the Data Scientist Skill Assessment Platform
            </p>
            <p class="silver-text" style="text-align: center;">
                Join our Data Science Talent Platform: Upload your resume to create a personalized profile, 
                then take our custom-built assessment designed to evaluate your skills in data analysis, 
                machine learning, statistics, and problem-solving. Based on your performance, 
                we'll showcase your strengths through a dynamic portfolio dashboard. Gain visibility with top recruiters,
                receive personalized feedback, and track your growth over time. Whether you're a beginner or an expert, 
                our platform helps you prove your capabilities and connect with real opportunities in the data science field.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(
            "<p class='silver-text' style='text-align: center; font-size: 18px;'>What We Assess:</p>", 
            unsafe_allow_html=True
        )
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("""
            <ul class="silver-text">
                <li>Python & R proficiency</li>
                <li>Data processing skills</li>
                <li>Statistical knowledge</li>
                <li>Machine learning expertise</li>
            </ul>
            """, unsafe_allow_html=True)
            
        with col_b:
            st.markdown("""
            <ul class="silver-text">
                <li>Data visualization abilities</li>
                <li>Problem-solving approach</li>
                <li>SQL and database knowledge</li>
                <li>Big data technologies</li>
            </ul>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Start Assessment"):
            st.session_state['current_page'] = 'upload'
            st.rerun()


def upload_page():
    """
    Render the CV upload page of the DataSkillAI platform.

    This function presents a page where users can upload their CV (PDF or DOCX format) 
    to begin the assessment process. It includes:

    - A welcoming title and motivational description to encourage the user.
    - A file uploader component to accept the user's CV.
    - Upon successful upload:
        - Extracts data from the CV using `extract_resume_data`.
        - Stores extracted data and generated questions into Streamlit session state.
        - Navigates the user to the instructions page to begin the assessment.
    - A "Back to Home" button allowing users to return to the home page.

    Streamlit's layout and markdown components are used for styling and flow.
    """
    st.markdown("<h1 class='main-title'>Upload Your CV</h1>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([2, 6, 2])
    
    with col_b:
        st.markdown("""
            <div >
                <p class="silver-text" style="font-size: 25px; text-align: center;">
                    Take Your Test!!
                </p>
                <p class="silver-text" style="text-align: center;">
                Ready to stand out in Data Science? Upload your CV to begin your journey.
                Once submitted, you'll get access to our custom assessment designed to evaluate 
                your practical skills in data analysis, machine learning, and problem-solving. 
                Your performance will be reflected in a dynamic profile, helping recruiters see your 
                true potential. Let your work speak louder than words!
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(" ", type=['pdf', 'docx'])

        if uploaded_file:
            st.session_state['uploaded_cv'] = uploaded_file
            
            extracted = extract_resume_data(uploaded_file)
            st.session_state['extracted_data'] = extracted
            
            st.session_state['questions'] = generate_llm_questions(extracted,5)
            
            st.session_state['current_page'] = 'instructions_page'
            st.rerun()
            
        if st.button("Back to Home"):
            st.session_state['current_page'] = 'home'
            st.rerun()


def instructions_page():
    """
    Display the test instructions page for the DataSkillAI assessment.

    This page serves as a pre-test briefing for users. It performs the following:
    
    - Retrieves the user's name from the previously extracted CV data.
    - If no data is found (e.g., the user hasn't uploaded a CV), a warning is displayed.
    - Greets the user by name and provides detailed instructions about how the test will proceed:
        - Questions will be based on their CV.
        - Each response is voice-based and limited to 2 minutes.
        - Users must click "Start Recording" to answer and "Save and Next" to proceed.
        - They may skip questions if needed.
        - The microphone should be enabled and background noise minimized.
        - If the session crashes, the test can be restarted.
    - A "Start Test" button transitions the user to the test interface.

    The content is styled using HTML within Streamlit for better visual clarity.
    """
    extracted_data = st.session_state.get('extracted_data')
    if not extracted_data:
        st.warning("No resume data found. Please upload your CV first.")
        return

    st.markdown(
        f"<h1 class='main-title'>Welcome {extracted_data['name']}"
        f"<span style='color:#9d50ff'>..!</span></h1>", 
        unsafe_allow_html=True
    )
    st.markdown("<hr style='border: 1px solid gray;'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="instruction-text">
        <p style='color:#9d50ff'><strong>Please read the instructions carefully before you begin:</strong></p>
        <p>You will be presented with questions one by one based on your CV.</p>
        <p>Each question has a 2-minute time limit to answer via voice.</p>
        <p>Press <strong>Start Recording</strong> to begin speaking your answer.</p>
        <p>Press <strong>Save and Next</strong> for the next question.</p>
        <p>You can skip questions if you're not ready or don't know the answer.</p>
        <p>Make sure your microphone is active and permissions are enabled.</p>
        <p>If the test crashes or closes, you can start from the beginning.</p>
        <p style='margin-top: 30px; font-weight: bold;'>When you're ready, click the <em>Start Test</em> button below!</p>
        <p><strong>Be in a quiet plcae for Best Performance:</strong></p>
    </div>
""", unsafe_allow_html=True)

    if st.button("Start Test"):
        st.session_state['current_page'] = 'test_page'
        st.rerun()


def test_page():
    """
    Display the main test interface for the DataSkillAI assessment.

    This function presents a series of CV-based questions to the user, one at a time, and allows
    them to respond via audio recording. Key features and flow include:

    - Checks if resume data is available; warns and exits if missing.
    - Greets the user by name and displays the current question.
    - Initializes session state variables to manage answers, timing, and progress.
    - Provides voice-based question playback using text-to-speech.
    - Allows users to record audio responses via the `audiorecorder` component.
    - Displays a countdown timer (2 minutes) for each question:
        - If the timer expires, the question is skipped automatically.
        - If the user clicks "Save & Next", the recorded response is transcribed and scored.
    - Users can replay the question, skip it, or move forward manually.
    - At the end of the test:
        - A final score and percentage are displayed.
        - The user has the option to return to the homepage and reset the test.

    This function handles real-time interactivity, voice processing, and dynamic test progression 
    using Streamlit's session state and reruns for responsiveness.
    """
    extracted_data = st.session_state.get('extracted_data')
    if not extracted_data:
        st.warning("No resume data found. Please upload your CV first.")
        return
    
    st.markdown(
        f"<h1 class='main-title'>Welcome {extracted_data['name']} </h1>", 
        unsafe_allow_html=True
    )
    st.markdown("---")            
    
    questions = st.session_state.get('questions', [])
    
    if "answers" not in st.session_state:
        st.session_state.answers = []  

    if "transcripts" not in st.session_state:
        st.session_state.transcripts = []

    # Initialize session state
    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0
        st.session_state.last_display_time = datetime.now().isoformat()
        st.session_state.spoken = False

    current_time = datetime.now()
    last_time = datetime.fromisoformat(st.session_state.last_display_time)
    time_elapsed = current_time - last_time
    time_remaining = timedelta(minutes=2) - time_elapsed

    if st.session_state.question_index < len(questions):
        question = questions[st.session_state.question_index]
        
        # Auto-speak on first display
        if not st.session_state.spoken:
            text_to_speech(question)
            st.session_state.spoken = True

        st.markdown(f"""
        <div style='text-align: center; font-size: 24px; font-weight: bold; margin-top: 20px;'>
            {st.session_state.question_index + 1}. {question}
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Centered buttons and audio input
        col0, col1, col2, col3 = st.columns([1, 1, 6, 2])
        
        with col0:
            if st.button("Play Again", key=f"play_{st.session_state.question_index}"):
                text_to_speech(question)

        with col2:
            col2_1, col2_2 = st.columns([1, 1])
            with col2_1:
                audio = audiorecorder("Start Recording", "⏹ Stop Recording")
            with col2_2:
                if len(audio) > 0:
                    audio_file_path = save_audio_to_wav(audio)
                    audio_buffer = io.BytesIO()
                    audio.export(audio_buffer, format="wav")
                    
                    if st.button("Save & Next", key=f"save_{st.session_state.question_index}"):
                        transcript = transcribe_audio(audio_file_path)
                        score = score_check(transcript, question)
                        st.session_state.answers.append({
                            "question": question,
                            "transcript": transcript,
                            "score": score
                        })
                        st.session_state.question_index += 1
                        st.session_state.last_display_time = datetime.now().isoformat()
                        st.session_state.spoken = False
                        st.rerun()
                    
                    st.audio(audio_buffer, format="audio/wav")
        with col3:
            if st.button("⏭ Skip", key=f"next_{st.session_state.question_index}"):
                st.session_state.answers.append({
                    "question": question,
                    "transcript": " "
                })
                st.session_state.question_index += 1
                st.session_state.last_display_time = datetime.now().isoformat()
                st.session_state.spoken = False
                st.rerun()

        # Styled timer
        if time_elapsed < timedelta(minutes=2):
            seconds_left = int(time_remaining.total_seconds())
            mins, secs = divmod(seconds_left, 60)
            st.markdown(f"""
                <div class="big-timer">
                    {mins:02d}:{secs:02d}
                </div>
            """, unsafe_allow_html=True)
            
            time.sleep(1)
            st.rerun()
        else:
            # Auto-advance
            st.session_state.question_index += 1
            st.session_state.last_display_time = datetime.now().isoformat()
            st.session_state.spoken = False
            st.rerun()
    else:
        st.markdown(
            "<h1 class='silver-text'>Sucessfuly completed your test</h1>", 
            unsafe_allow_html=True
        )
        total_score, percentage = calculate_score(st.session_state.answers)
        st.markdown(f"""
            <div style='text-align: center; font-size: 28px; font-weight: bold; color: 9d50ff; margin-top: 20px;'>
                Score: {total_score} <br>
                Rating: {percentage:.1f}%
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")

        if st.button("Return to Home"):
            extracted_data = st.session_state.get("extracted_data", {})
            candidate_name = extracted_data.get("name", "Unknown")
            email = extracted_data.get("email", "unknown@example.com")
            total_score, percentage = calculate_score(st.session_state.answers)
            total = len(st.session_state.answers) * 10
            
            conn_params = {
                "host": os.getenv("DB_HOST"),
                "database": os.getenv("DB_NAME"),
                "user": os.getenv("DB_USER"),
                "password": os.getenv("DB_PASSWORD"),
                "port": os.getenv("DB_PORT")
            }
             
            save_candidate_results(candidate_name, email, total_score, total, percentage, conn_params)

            
            st.session_state.current_page = "home"
            st.session_state.question_index = 0
            st.session_state.spoken = False
            st.session_state.last_display_time = datetime.now().isoformat()
            st.session_state.answers = []
            st.rerun()


def main():
    """Main function to control page navigation."""
    if st.session_state['current_page'] == 'home':
        home_page()
    elif st.session_state['current_page'] == 'upload':
        upload_page()
    elif st.session_state.get('current_page') == 'instructions_page':
        instructions_page()
    elif st.session_state.get('current_page') == 'test_page':
        test_page()


if __name__ == "__main__":
    main()