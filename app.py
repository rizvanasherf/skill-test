import streamlit as st
import time
import io
from datetime import datetime, timedelta
from audiorecorder import audiorecorder
from utils import extract_resume_data,generate_llm_questions,uniquequestion
from utils import transcribe_audio,save_audio_to_wav,text_to_speech



# Set page configuration
st.set_page_config(
    page_title="DataSkillTest.AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS for black background and silver/gray text
with open("asset/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state variables
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'

if 'uploaded_cv' not in st.session_state:
    st.session_state['uploaded_cv'] = None

# ----------------- PAGE FUNCTIONS -----------------

# Home Page
def home_page():
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
        
        st.markdown("<p class='silver-text' style='text-align: center; font-size: 18px;'>What We Assess:</p>", unsafe_allow_html=True)
        
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
    st.markdown("<h1 class='main-title'>Upload Your CV</h1>", unsafe_allow_html=True)
    col_a, col_b,col_c = st.columns([2,6,2])
    
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
            st.success("CV uploaded successfully!")
            
            extracted = extract_resume_data(uploaded_file)
            st.session_state['extracted_data'] = extracted
            
            st.session_state['current_page'] = 'test_page'
            st.rerun()
            
        if st.button("Back to Home"):
            st.session_state['current_page'] = 'home'
            st.rerun()
            
def test_page():
    
    extracted_data = st.session_state.get('extracted_data')
    if not extracted_data:
        st.warning("No resume data found. Please upload your CV first.")
        return
    
    st.markdown(f"<h1 class='main-title'>Welcome {extracted_data['name']} </h1>", unsafe_allow_html=True)
    st.markdown("---")            
    st.markdown("### Generated Interview Questions:")
    st.json(extracted_data)

    questions = uniquequestion('abc')
    
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
    time_remaining = timedelta(minutes=1) - time_elapsed

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
        col0, col1, col2, col3 = st.columns([1,1, 6, 2])
        
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
                    st.audio(audio_buffer, format="audio/wav")

                    if st.button("Save & Next", key=f"save_{st.session_state.question_index}"):
                        transcript = transcribe_audio(audio_file_path)
                        st.session_state.answers.append({
                            "question": question,
                            "transcript": transcript
                        })
                        st.session_state.question_index += 1
                        st.session_state.last_display_time = datetime.now().isoformat()
                        st.session_state.spoken = False
                        st.rerun()

        with col3:
            if st.button("⏭ Skip", key=f"next_{st.session_state.question_index}"):
                st.session_state.question_index += 1
                st.session_state.last_display_time = datetime.now().isoformat()
                st.session_state.spoken = False
                st.rerun()

        # Styled timer
        if time_elapsed < timedelta(minutes=1):
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
        st.write("All questions completed!")
        for i, answer in enumerate(st.session_state.answers):
            st.markdown(f"**Q{i+1}: {answer['question']}**")
            st.markdown(f"**Your Answer:** {answer['transcript']}**")
            st.markdown("---")

def main():
    if st.session_state['current_page'] == 'home':
        home_page()
    elif st.session_state['current_page'] == 'upload':
        upload_page()
    elif st.session_state.get('current_page') == 'test_page':
        test_page()



if __name__ == "__main__":
    main()
