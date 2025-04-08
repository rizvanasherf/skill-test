import pdfplumber 
import os
import docx
import re
from collections import defaultdict
from openai import OpenAI
import time
from dotenv import load_dotenv
import tempfile
from pydub import AudioSegment
import wave
import random
import whisper
import io
import pyttsx3
import threading



load_dotenv() 

_GLOBAL_LLM_CLIENT = None

whisper_model = whisper.load_model("base")

def save_audio_to_wav(audio_segment):
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)

    # Save the audio to a temporary .wav file
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with open(temp_wav.name, 'wb') as f:
        f.write(wav_io.read())

    return temp_wav.name

def transcribe_audio(audio_path):
    try:
        result = whisper_model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        return f"Transcription failed: {str(e)}"
    finally:
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)
    
def configure_llm():
    """Configure and validate API key for LLM.
    
    Returns:
    - str: The Open AI client 
    
    Raises:
    - ValueError: If the API key is not found in the environment variables.
    """
    global _GLOBAL_LLM_CLIENT
    
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found")
    _GLOBAL_LLM_CLIENT = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )
    return _GLOBAL_LLM_CLIENT
    
def call_grok(prompt, max_retries=3):
    """
    Synchronous Grok API call with retries.

    Args:
    - prompt (str): The prompt to send.
    - max_retries (int): Number of retry attempts on failure.

    Returns:
    - str: Response content or error message.
    """
    global _GLOBAL_LLM_CLIENT

    if _GLOBAL_LLM_CLIENT is None:
        try:
            configure_llm()
        except Exception as config_error:
            return f"Configuration Error: {config_error}"

    payload = {
        "model": "grok-2-latest",
        "messages": [
            {"role": "system", "content": "You are an AI assistant"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }

    for attempt in range(max_retries):
        try:
            response = _GLOBAL_LLM_CLIENT.chat.completions.create(**payload)
            if response and response.choices:
                return response.choices[0].message.content
            else:
                continue

        except Exception as e:
            print(f"API Call Error (Attempt {attempt + 1}/{max_retries}): {e}")
            if hasattr(e, 'http_status') and e.http_status == 429:
                print("Rate limit exceeded. Retrying with backoff.")
                time.sleep(2 ** attempt)
                continue
            elif hasattr(e, 'type') and e.type == 'invalid_request_error':
                print("Invalid request. Please check the payload.")
                break

    return "Failed to get a response after multiple attempts."

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_keywords(text):
    data = defaultdict(list)
    
    text_lower = text.lower()
    
    # Extract contact information first
    # Name extraction - typically appears at the top of resume
    name_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
    name_match = re.search(name_pattern, text.strip())
    if name_match:
        data['name'] = name_match.group(0).strip()
    
    # Email extraction - look for standard email format
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, text)
    if email_match:
        data['email'] = email_match.group(0).strip()
    
    # Phone number extraction - handles various formats
    # This pattern covers common formats like (123) 456-7890, 123-456-7890, 123.456.7890, etc.
    phone_patterns = [
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US/Canada: (123) 456-7890 or 123-456-7890
        r'\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # International: +1 123-456-7890
        r'\+\d{1,3}[-.\s]?\d{10,12}',  # International compact: +11234567890
        r'\d{10,12}'  # Simple 10-12 digit number
    ]
    
    for pattern in phone_patterns:
        phone_match = re.search(pattern, text)
        if phone_match:
            data['phone'] = phone_match.group(0).strip()
            break
    
    # LinkedIn URL extraction (common in modern resumes)
    linkedin_pattern = r'linkedin\.com/in/[A-Za-z0-9_-]+'
    linkedin_match = re.search(linkedin_pattern, text)
    if linkedin_match:
        data['linkedin'] = linkedin_match.group(0).strip()
    
    # Original skills extraction code
    ds_ml_keywords = {
        'programming': ['python', 'r', 'sql', 'scala', 'julia', 'java', 'c++'],
        'ml_frameworks': ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'spark ml', 'mxnet'],
        'data_processing': ['pandas', 'numpy', 'pyspark', 'dask', 'modin', 'apache beam'],
        'data_visualization': ['matplotlib', 'seaborn', 'plotly', 'ggplot', 'tableau', 'power bi'],
        'big_data': ['hadoop', 'hive', 'spark', 'kafka', 'airflow', 'databricks'],
        'ml_techniques': [
            'machine learning', 'deep learning', 'supervised learning', 
            'unsupervised learning', 'reinforcement learning', 'transfer learning',
            'natural language processing', 'computer vision', 'time series',
            'feature engineering', 'model evaluation', 'hyperparameter tuning'
        ],
        'nlp': ['nltk', 'spacy', 'gensim', 'transformers', 'bert', 'gpt', 'word2vec'],
        'cv': ['opencv', 'pil', 'yolo', 'faster r-cnn', 'image processing'],
        'deployment': ['flask', 'django', 'fastapi', 'docker', 'kubernetes', 'aws', 'azure', 'gcp']
    }
    
    for category, keywords in ds_ml_keywords.items():
        found_keywords = [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]
        if found_keywords:
            data['skills'].extend(found_keywords)
            data[f'ds_ml_{category}'] = found_keywords
    
    education_pattern = r'(education|academic background|qualifications|degrees?)(.*?)(?=(work|experience|projects|skills|$)|\n\n)'
    education_match = re.search(education_pattern, text_lower, re.DOTALL | re.IGNORECASE)
    if education_match:
        data['education'] = education_match.group(2).strip()
    
    projects_pattern = r'(projects|personal projects|work samples|selected projects)(.*?)(?=(work|experience|education|skills|$)|\n\n)'
    projects_match = re.search(projects_pattern, text_lower, re.DOTALL | re.IGNORECASE)
    if projects_match:
        projects_text = projects_match.group(2).strip()
        data['projects_raw'] = projects_text
        
        project_items = re.split(r'\n\s*\d+\.|\n\s*[-•*]|\n\s*(?=[A-Z][a-z])', projects_text)
        project_items = [p.strip() for p in project_items if p.strip()]
        data['projects'] = project_items
        
        ml_projects = []
        for project in project_items:
            if any(re.search(r'\b' + re.escape(kw) + r'\b', project.lower()) 
               for kw in ds_ml_keywords['ml_techniques'] + ds_ml_keywords['ml_frameworks']):
                ml_projects.append(project)
        data['ml_projects'] = ml_projects
    
    experience_pattern = r'(experience|work history|employment)(.*?)(?=(projects|education|skills|$)|\n\n)'
    experience_match = re.search(experience_pattern, text_lower, re.DOTALL | re.IGNORECASE)
    if experience_match:
        exp_text = experience_match.group(2).strip()
        data['experience_raw'] = exp_text
        
        ds_exp = []
        for line in exp_text.split('\n'):
            if any(re.search(r'\b' + re.escape(kw) + r'\b', line.lower()) 
                   for kw in ds_ml_keywords['ml_techniques'] + ['data scientist', 'machine learning', 'ml engineer']):
                ds_exp.append(line.strip())
        data['ds_experience'] = ds_exp
    
    # Try to extract a summary or objective statement if present
    summary_pattern = r'(summary|objective|profile|about me)(.*?)(?=(education|experience|projects|skills|$)|\n\n)'
    summary_match = re.search(summary_pattern, text_lower, re.DOTALL | re.IGNORECASE)
    if summary_match:
        data['summary'] = summary_match.group(2).strip()
    
    return dict(data)

def extract_resume_data(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == 'docx':
            text = extract_text_from_docx(uploaded_file)
        else:
            return {"error": "Unsupported file format"}

        extracted_info = extract_keywords(text)
        
        if 'skills' in extracted_info:
            extracted_info['skills_count'] = len(extracted_info['skills'])
        if 'projects' in extracted_info:
            extracted_info['projects_count'] = len(extracted_info['projects'])
        if 'ml_projects' in extracted_info:
            extracted_info['ml_projects_count'] = len(extracted_info['ml_projects'])
        
        return extracted_info
    except Exception as e:
        return {"error": str(e)}

def generate_llm_questions(extracted_data, total_questions=None):
    """
    Generate interview questions with varying difficulty levels based on extracted resume data.
    60% technical skills (evenly distributed across difficulty levels)
    20% projects-based questions
    20% experience-based questions
    """
    context = {
        'skills': extracted_data.get('skills', []),
        'projects': extracted_data.get('ml_projects', []),
        'experience': extracted_data.get('ds_experience', [])
    }
    
    # Categorize skills by domain to create more specific questions
    skill_categories = {
        'programming': [s for s in context['skills'] if s in extracted_data.get('ds_ml_programming', [])],
        'ml_frameworks': [s for s in context['skills'] if s in extracted_data.get('ds_ml_ml_frameworks', [])],
        'data_processing': [s for s in context['skills'] if s in extracted_data.get('ds_ml_data_processing', [])],
        'ml_techniques': [s for s in context['skills'] if s in extracted_data.get('ds_ml_ml_techniques', [])]
    }
    
    all_questions = []
    
    # Generate technical questions with difficulty levels
    difficulty_levels = ["beginner", "intermediate", "advanced"]
    tech_question_count = int(total_questions * 0.8)
    questions_per_level = tech_question_count // len(difficulty_levels)
    
    for difficulty in difficulty_levels:
        for category, skills in skill_categories.items():
            if skills:
                sampled_skills = random.sample(skills, k=random.randint(1, len(skills)))
                
                # Create targeted prompts for different difficulty levels
                if difficulty == "beginner":
                    prompt = f"""Generate {questions_per_level} beginner-level technical interview questions for a Data Science role.
                        Focus on foundational concepts and core principles related to the following skills: {', '.join(sampled_skills)}.
                        Questions should assess basic understanding and be appropriate for entry-level candidates.
                        Format the output as plain questions only, without numbering and have to be with a question format, without providing answers."""

                elif difficulty == "intermediate":
                    prompt = f"""Generate {questions_per_level} intermediate-level technical interview questions for a Data Science role.
                        Focus on practical application, real-world use cases, and moderate complexity across the following skills: {', '.join(sampled_skills)}.
                        Questions should evaluate hands-on experience, applied knowledge, and problem-solving ability.
                        Format the output as plain questions only, without numbering and have to be with a question format, without providing answers."""
                                            
                else:  # advanced
                   prompt = f"""Generate {questions_per_level} advanced-level technical interview questions for a Data Science role.
                        Focus on deep expertise, edge cases, and complex real-world scenarios involving the following skills: {', '.join(sampled_skills)}.
                        The questions should assess expert-level understanding, including optimization techniques, theoretical depth, and practical implementation challenges.
                        Format the output as plain questions only, without numbering and have to be with a question format, without providing answers."""
                                    
                response = call_grok(prompt)
                if not response.startswith("Failed"):
                    questions = [q.strip() for q in response.split('\n') if q.strip() and q.strip().endswith('?')]
                    all_questions.extend(questions[:questions_per_level])
    
    # Generate project-based questions
    if context['projects']:
        proj_question_count = int(total_questions * 0.1)
        project_prompt = f"""Based on these projects: {' | '.join(context['projects'][:3])},
        generate {proj_question_count} project-based interview questions that:
        1. Probe the candidate's technical decision-making process
        2. Investigate challenges faced and how they were overcome
        3. Explore the depth of their understanding of the technologies used
        Format the output as plain questions only, without numbering and have to be in a question format without answers."""
        
        response = call_grok(project_prompt)
        if not response.startswith("Failed"):
            questions = [q.strip() for q in response.split('\n') if q.strip() and q.strip().endswith('?')]
            all_questions.extend(questions[:proj_question_count])
    
    # Generate experience-based questions
    if context['experience']:
        exp_question_count = int(total_questions * 0.1)
        exp_prompt = f"""Based on this experience: {' | '.join(context['experience'][:3])},
        generate {exp_question_count} experience-based interview questions that:
        1. Assess the candidate's problem-solving in real-world scenarios
        2. Evaluate their ability to work with stakeholders and communicate results
        3. Probe their knowledge of data science workflows and best practices
        Format the output as plain questions only, without numbering and have to be in a question format without answers."""
        
        response = call_grok(exp_prompt)
        if not response.startswith("Failed"):
            questions = [q.strip() for q in response.split('\n') if q.strip() and q.strip().endswith('?')]
            all_questions.extend(questions[:exp_question_count])
    
    # Ensure we have enough questions
    if len(all_questions) < total_questions:
        # Generate general data science questions to fill the gap
        remaining = total_questions - len(all_questions)
        general_prompt = f"""Generate {remaining} general data science interview questions of varying difficulty.
        Include questions about methodology, model evaluation, and industry best practices.
        Format the output as plain questions only, without numbering and have to be in a question format without answers."""
        
        response = call_grok(general_prompt)
        if not response.startswith("Failed"):
            questions = [q.strip() for q in response.split('\n') if q.strip() and q.strip().endswith('?')]
            all_questions.extend(questions[:remaining])
    
    # Remove duplicates and ensure proper formatting
    seen = set()
    unique_questions = []
    
    for q in all_questions:
        # Remove numbering at the beginning of questions
        clean_q = re.sub(r'^\d+[\.\)\-]\s*', '', q)
        # Get core question text for deduplication
        core_q = re.sub(r'^\[[A-Z]+\]\s*', '', clean_q).lower()
        
        if core_q not in seen and len(core_q) > 20:  # Avoid very short questions
            seen.add(core_q)
            unique_questions.append(clean_q)
    
    # Shuffle questions to mix difficulty levels and types
    random.shuffle(unique_questions)
    
    return unique_questions[:total_questions]

def uniquequestion(text):
    dummy_questions = [
    "What is the game loop? Explain its components",
    "What is collision detection and how is it implemented",
    "What are coroutines and where would you use them in Unity"
]
    return dummy_questions

def text_to_speech(text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run).start()
    
# def Progress_checking()