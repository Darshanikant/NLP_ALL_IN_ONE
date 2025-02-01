# libraries
import streamlit as st
import numpy as np
import spacy
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator 
import pandas as pd
import base64  # For background image handling
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import os
import speech_recognition as sr
from gtts import gTTS
import joblib
import language_tool_python
import pyttsx3


# Set background image
def set_bg():
    page_bg_img = f'''
    <style>
    .stApp {{
        background: url("https://media.licdn.com/dms/image/v2/D4D12AQGklLlO96-8Ig/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1721238552279?e=2147483647&v=beta&t=XyPoWeFiwDF0eItKlqL56DZKjcJsoyn-lm6s4Vfmuy4");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Apply background
set_bg()

# Sidebar Navigation
st.sidebar.title("📌 App Features")
st.sidebar.info("Select a task below to explore NLP functionalities.")

st.sidebar.markdown("---")

task = st.sidebar.radio("Select Task", [
    "Home", "Sentiment Analysis", "Named Entity Recognition", 
    "Language Translation", "Text Similarity", "Keyword Extraction", "Word Cloud Generator","Speech-to-Text",
    "Text-to-Speech","Grammar & Spell Check"
])

st.sidebar.markdown("---")
st.sidebar.write("© Developed by **Darshanikanta 2025**")

# Home Page
if task == "Home":
    st.title("✨ NLP All-in-One Application ✨")
    st.markdown("### About Me 🧑‍💻")
    st.write("**Name:** Darshanikanta Behera🤖")  
    st.write("**Role:** Data Scientist / NLP Enthusiast 📊 / 🤖")  
    st.write("**Email:** darshanikanta@gmail.com 📧") 
    st.write("**Github:** [GitHub](https://github.com/Darshanikant) 🔗")
    st.write("**Linkedln:** [LinkedIn](https://www.linkedin.com/in/darshanikanta-behera-b0377b296/) 🔗")
    st.markdown("### About Project 📚")
    st.markdown("""
    **Welcome to the ultimate NLP application! 🚀**
    This application provides various NLP functionalities including:
    
    - **Sentiment Analysis** 😊😞 - Analyze the emotion behind the text.
    - **Named Entity Recognition (NER)** 🏢📅 - Identify entities like names, places, and organizations.
    - **Language Translation** 🌍 - Translate text into different languages.
    - **Text Similarity** 🔍 - Measure the similarity between two texts.
    - **Keyword Extraction** 🏷️ - Extract important words from the text.
    - **Word Cloud Generator** ☁️ - Generate a visual representation of word frequencies.
    - **Speech-to-Text** 🗣️ - Convert speech to text.
    - **Text-to-Speech** 🗣️ - Convert text to speech.
    - **Grammar & Spell Check** 📝 - Check grammar and spell errors.
    
    Choose a task from the sidebar and get started!
    """)
      

# Sentiment Analysis
elif task == "Sentiment Analysis":
    st.subheader("🔹 Sentiment Analysis")
    st.markdown("""This section analyzes the sentiment of the given text and determines whether the text expresses a positive, negative, or neutral emotion. It uses a pre-trained model from Hugging Face's transformers library.""")
    text = st.text_area("Enter text for sentiment analysis:", "I love this product! It's amazing.")
    if st.button("Analyze Sentiment"):
        try:
            sentiment_analyzer = pipeline("sentiment-analysis")
            result = sentiment_analyzer(text)
            sentiment_label = result[0]['label']
            confidence = result[0]['score']
            sentiment_to_emoji = {"POSITIVE": "😊", "NEGATIVE": "😞", "NEUTRAL": "😐"}
            st.success(f"**Sentiment:** {sentiment_label} {sentiment_to_emoji.get(sentiment_label, '🤔')}")
            st.info(f"**Confidence:** {confidence:.2f}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Named Entity Recognition (NER)
elif task == "Named Entity Recognition":
    st.subheader("🔹 Named Entity Recognition (NER)")
    st.markdown("""NER identifies and categorizes proper names, places, dates, and more from the given text using the SpaCy library.""")
    text = st.text_area("Enter text for NER:", "Elon Musk is the CEO of Tesla.")
    if st.button("Extract Entities"):
        

        from spacy.cli import download

# Try to load the SpaCy model, and if it fails, download it
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
    
        doc = nlp(text)
        for ent in doc.ents:
            st.write(f"{ent.text} ({ent.label_})")

# Language Translation
elif task == "Language Translation":
    st.subheader("🔹 Language Translation")
    st.markdown("""This feature translates input text into multiple target languages using Google Translate API.""")
    
    language_dict = {
        "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "bs": "Bosnian",
        "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German",
        "el": "Greek", "en": "English", "eo": "Esperanto", "es": "Spanish", "et": "Estonian",
        "fi": "Finnish", "fr": "French", "gu": "Gujarati",  "hi": "Hindi",
        "hr": "Croatian", "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian", "is": "Icelandic",
        "it": "Italian", "ja": "Japanese", "jw": "Javanese", "km": "Khmer", "kn": "Kannada",
        "ko": "Korean", "la": "Latin", "lv": "Latvian", "mk": "Macedonian", "ml": "Malayalam",
        "mr": "Marathi", "my": "Myanmar (Burmese)", "ne": "Nepali", "nl": "Dutch", "no": "Norwegian",
        "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "si": "Sinhala",
        "sk": "Slovak", "sq": "Albanian", "sr": "Serbian", "su": "Sundanese", "sv": "Swedish",
        "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai", "tl": "Filipino",
        "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "zh-CN": "Chinese"
    }
    
    text = st.text_area("Enter text to translate:", "Hello, how are you?")
    target_language = st.selectbox("Select Target Language", [f"{code} : {name}" for code, name in language_dict.items()])
    
    if st.button("Translate"):
        try:
            lang_code = target_language.split(" : ")[0]
            translator = Translator()
            translation = translator.translate(text, dest=lang_code)
            st.success(f"**Translated Text:** {translation.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# The rest of the code remains unchanged

# Text Similarity
elif task == "Text Similarity":
    st.subheader("🔹 Text Similarity")
    st.markdown("""This feature calculates how similar two pieces of text are using TF-IDF vectorization and cosine similarity.""")
    text1 = st.text_area("Enter first text:", "Hello, I am a user.")
    text2 = st.text_area("Enter second text:", "Hello, I am a user.")
    if st.button("Calculate Similarity"):
        if text1 and text2:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            st.success(f"**Similarity Score:** {similarity_score:.4f}")
        else:
            st.warning("Please enter both texts.")

# Word Cloud Generator
elif task == "Word Cloud Generator":
    st.subheader("🔹 Word Cloud Generator ☁️")
    st.markdown("""This feature generates a word cloud, a visual representation of text frequency, from the provided input.""")
    text = st.text_area("Enter text to generate a word cloud:", "Data Science is the Topic now Trained in the AI Model.")
    if st.button("Generate Word Cloud"):
        if text:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

            img_bytes = BytesIO()
            wordcloud.to_image().save(img_bytes, format='PNG')
            img_bytes.seek(0)
            b64 = base64.b64encode(img_bytes.read()).decode()
            href = f'<a href="data:file/png;base64,{b64}" download="wordcloud.png">Download Word Cloud</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("Please enter text to generate a word cloud!")




# Text-to-Speech with Male & Female Voice
elif task == "Text-to-Speech":
    st.subheader("🔹 Text-to-Speech 🔊")
    st.markdown("Convert text into speech and download the audio file.")

    text = st.text_area("Enter text to convert to speech:", "Hello! Welcome to NLP.")

    # Dropdown for voice selection
    voice_option = st.selectbox("Choose Voice:", ["Male", "Female"])
    

    # Function to convert text to speech
    def text_to_speech(text, gender="Male"):
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')

        # Set male or female voice based on user selection
        if gender == "Male":
            engine.setProperty('voice', voices[0].id)  # Typically Male
        else:
            engine.setProperty('voice', voices[1].id)  # Typically Female

        engine.save_to_file(text, "output.mp3")
        engine.runAndWait()

    # Convert text to speech on button click
    if st.button("Convert to Speech"):
        if text.strip():
            text_to_speech(text, voice_option)
            st.audio("output.mp3")
            
            # Provide download button for the audio
            with open("output.mp3", "rb") as audio_file:
                st.download_button(label="Download Audio", data=audio_file, file_name="speech.mp3", mime="audio/mp3")
        else:
            st.error("Please enter text to convert.")

# Speech-to-Text (STT)
elif task == "Speech-to-Text":
    st.subheader("🔹 Speech-to-Text (STT)")
    st.markdown("This feature allows you to convert speech to text either via microphone or by uploading an audio file.")
    recognizer = sr.Recognizer()
    
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
    if st.button("Transcribe from Audio File") and audio_file:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            st.success(f"**Transcribed Text:** {text}")
    
    if st.button("Record from Microphone"):
        with sr.Microphone() as source:
            st.info("Recording... Speak now!")
            audio_data = recognizer.listen(source)
            text = recognizer.recognize_google(audio_data)
            st.success(f"**Transcribed Text:** {text}")

# Grammar & Spell Check
elif task == "Grammar & Spell Check":
    st.subheader("🔹 Grammar & Spell Check")
    st.markdown("""
    This feature checks for spelling and grammatical errors using **LanguageTool**.
    """)
    
    tool = language_tool_python.LanguageTool("en-US")
    input_text = st.text_area("Enter text for correction:", "This is an example sentence with erors.")
    if st.button("Check Grammar & Spelling"):
        corrected_text = tool.correct(input_text)
        st.success(f"Corrected Text: {corrected_text}")
# Keyword Extraction
elif task == "Keyword Extraction":
    st.subheader("🔹 Keyword Extraction")
    st.markdown("""
    This feature extracts the most relevant keywords from the given text using **TF-IDF Vectorization**.
    """)
    
    input_text = st.text_area("Enter text to extract keywords:", "Artificial Intelligence is transforming the world of technology.")
    keyword_number=st.number_input("Enter The number of keywords:", value=5)

    def extract_keywords(text, n=keyword_number):
        vectorizer = TfidfVectorizer(stop_words="english", max_features=10)
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
        top_keywords = feature_array[tfidf_sorting][:n]
        return top_keywords

    if st.button("Extract Keywords"):
        if input_text.strip():
            keywords = extract_keywords(input_text)
            st.success(f"Top Keywords: {', '.join(keywords)}")
        else:
            st.error("Please enter some text to extract keywords.")

