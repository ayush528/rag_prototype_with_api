# RAG Prototype with API

This repo contains a prototype of a Retrieval-Augmented Generation (RAG) system with API support.

## 📔 Open the Notebook

You can run the notebook directly in the browser using Google Colab or Binder:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayush528/rag_prototype_with_api/blob/master/rag_prototype_with_api.ipynb)

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ayush528/rag_prototype_with_api/master?filepath=rag_prototype_with_api.ipynb)



# Overview
This prototype implements a multilingual voice-based Retrieval-Augmented Generation (RAG) system designed for visually impaired users. It enables voice-driven content search and interaction, delivering responses in both text and audio formats for accessibility. The system uses open-source tools for speech-to-text (STT), semantic retrieval, and text-to-speech (TTS), with response generation via Google’s Gemini API (accessed through an OpenAI-compatible client). Optimized for Google Colab, it supports on-premise deployment considerations by leveraging local models where possible.
Pipeline Design
The pipeline processes a voice query from an uploaded audio file (input.wav), retrieves relevant documents from a corpus, generates a response, and outputs it as text and audio. The steps are:

# Voice Input (STT):

Tool: OpenAI Whisper (tiny model).
Function: Transcribes audio input (input.wav, 16kHz WAV) into text.
Details: Supports multilingual transcription, currently set to English for simplicity. Audio is sampled at 16kHz for compatibility with Whisper.


# Document Retrieval:

Embedding Model: SentenceTransformers (all-MiniLM-L6-v2).
Vector Store: FAISS (FlatL2 index).
Function: Embeds a corpus of documents and the transcribed query into 384-dimensional vectors, retrieving the top-2 most semantically similar documents using FAISS.
Details: The corpus is stored in corpus.txt, with documents sourced from a book excerpt on web accessibility (e.g., screen readers, WCAG 2.1).


# Response Generation:

Tool: Google Gemini API (gemini-2.5-flash-preview-04-17, via OpenAI client).
Function: Generates a natural language response using the transcribed query and retrieved documents as context.
Details: Uses an OpenAI-compatible client with a custom base_url (https://generativelanguage.googleapis.com/v1beta/openai/). The prompt includes a system instruction (“You are a helpful assistant”) and the query with context.


Output (TTS and Text):

Tool: Coqui TTS (tts_models/en/ljspeech/tacotron2-DDC).
Function: Converts the API response into audio (output.wav) and saves the text (response.txt).
Details: Audio is generated in English; text output ensures accessibility for screen readers or braille displays.



# Tools Used

Whisper: Multilingual STT model for transcribing voice queries.
SentenceTransformers: Generates dense embeddings for semantic retrieval.
FAISS: Efficient vector search for retrieving relevant documents.
Gemini API: External API for response generation, accessed via OpenAI client.
Coqui TTS: Multilingual TTS for audio output.
Librosa/Soundfile: Processes audio files, avoiding PortAudio dependencies.
Scipy: Optional for local audio recording with scipy.io.wavfile.write.
OpenAI Client: Facilitates API calls to Gemini’s endpoint.

# Implementation Notes

Corpus:
The corpus is a ~550-word excerpt from a hypothetical book chapter, “The World Wide Web and Accessibility,” split into 8 documents. Topics include screen readers, semantic HTML, keyboard navigation, and WCAG 2.1, aligning with the prototype’s accessibility focus.
Example document: “Screen readers are essential assistive technologies for visually impaired users. They convert digital text into synthesized speech or braille output, allowing users to navigate websites, read documents, and interact with applications.”
Stored in corpus.txt for persistence and retrieval.


Voice Input:
Google Colab does not support direct audio recording, so users upload a 16kHz WAV file (input.wav) containing the query (e.g., “How do screen readers help visually impaired users?”).
An optional record_audio function is provided for local use, requiring PortAudio and sounddevice.


Gemini API:
Accessed via openai client with a custom base_url and model (gemini-2.5-flash-preview-04-17). Requires a user-provided API key.
The API call uses a messages structure with system and user roles, incorporating retrieved documents as context.


On-Premise Consideration:
Whisper, SentenceTransformers, FAISS, and Coqui TTS run locally, supporting on-premise deployment.
The Gemini API is external, conflicting with on-premise goals. Replace with a local LLM (e.g., LLaMA) for production.


Multilingual Support:
English is used for simplicity. Whisper supports multilingual transcription by setting language=None in voice_to_text.


Dependency Management:
Pinned versions (numpy==1.23.5, transformers==4.38.2) resolve compatibility issues (e.g., numpy.dtypes errors).
No PortAudio/sounddevice dependencies in Colab, ensuring smooth execution.



# Setup Instructions

Environment:
Open rag_prototype_with_api.ipynb in Google Colab.


Dependencies:
Run the first cell to install pinned packages:!pip install numpy==1.23.5
!pip install transformers==4.38.2
!pip install git+https://github.com/openai/whisper.git
!pip install sentence-transformers
!pip install faiss-cpu
!pip install tts
!pip install soundfile
!pip install librosa
!pip install requests
!pip install scipy
!pip install openai




API Key:
Obtain a Gemini API key from Google Cloud.
Replace "YOUR_GEMINI_API_KEY" in the notebook with your key.


Input Audio:
Record a query (e.g., “How do screen readers help visually impaired users?”) using Audacity or an online tool.
Upload to Colab via the “Files” tab.


Corpus:
The notebook defines a book excerpt corpus (8 documents) in the documents list, written to corpus.txt.


Execution:
Run all cells to:
Transcribe input.wav.
Retrieve relevant documents.
Generate a response via Gemini API.
Save response.txt and output.wav.


Play output.wav in Colab to verify audio output.


Outputs:
corpus.txt: Book excerpt documents.
response.txt: Text response from Gemini API.
output.wav: Audio response via Coqui TTS.



# Test Case

Corpus: 8 documents from a ~550-word book excerpt on web accessibility, covering screen readers, semantic HTML, keyboard navigation, etc.
Query: “How do screen readers help visually impaired users?” (recorded in input.wav).
Expected Outcomes:
Transcription: “How do screen readers help visually impaired users?”.
Retrieved Document: “Screen readers are essential assistive technologies for visually impaired users. They convert digital text into synthesized speech or braille output, allowing users to navigate websites, read documents, and interact with applications.”
Response: “Screen readers help visually impaired users by converting digital text into speech or braille, enabling them to navigate websites, read documents, and use applications.”
Outputs: response.txt and output.wav match the response.


Verification: Check printed outputs (Transcribed Query, Retrieved Documents, Generated Response) and play output.wav.

# Limitations

Colab Audio Input: Requires uploaded input.wav due to lack of microphone access in Colab.
Whisper Model: The tiny model may have lower accuracy for non-English or noisy audio. Use larger models (e.g., base) for better performance.
External API: The Gemini API is cloud-based, conflicting with on-premise goals. A local LLM is needed for production.
Corpus Size: The book excerpt (~550 words, 8 documents) is small. Scaling to larger corpora requires optimized FAISS indexing.
Multilingual Support: Limited to English; full multilingual support requires language detection in Whisper.

# Future Improvements

Multilingual Support: Enable language detection in Whisper (language=None) for multilingual queries.
Local LLM: Replace Gemini API with a local model (e.g., LLaMA) for on-premise deployment.
Braille Output: Integrate python-braille for braille display compatibility.
Corpus Expansion: Support larger corpora with hierarchical FAISS indexing.
Audio Quality: Use a higher-quality Whisper model or enhance Coqui TTS voices.


Uploading input.wav (e.g., “How do screen readers help visually impaired users?”).
Running the Colab notebook with the book excerpt corpus.
Displaying printed outputs: transcribed query, retrieved documents, generated response.
Playing output.wav to verify audio output.
Showing response.txt and corpus.txt.Use a screen recording tool (e.g., OBS Studio) and save as demo_video.mp4. Upload to the GitHub repository.

# Repository Structure

rag_prototype_with_api.ipynb: Colab notebook with RAG pipeline.
corpus.txt: Book excerpt corpus.
response.txt: Generated text response.
input.wav: Sample query audio.
output.wav: Generated audio response.
README.md: Project overview and setup.
docs/rag_system_documentation.md: Detailed pipeline documentation.
demo_video.mp4: Demo video.




