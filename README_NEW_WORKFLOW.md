# Meeting Transcription Tool - New Workflow

## Overview

This tool has been updated to use **ElevenLabs** for high-quality transcription and **Mistral** for AI processing, replacing the previous Deepseek integration.

## New Workflow: Video + Teams Transcript

### 🎥 Step 1: Upload Video
- Upload the meeting video file (MP4, AVI, MOV, MKV, WEBM)
- The video will be processed by ElevenLabs for accurate transcription

### 📝 Step 2: Provide Teams Transcript
- Copy and paste the Teams transcript (which contains speaker names)
- This provides speaker identification for the accurate transcription

### 🤖 Step 3: AI Combination
- Mistral AI combines the accurate transcription with speaker names
- Results in a high-quality transcript with proper speaker identification

### 📊 Step 4: Information Extraction
- Extract structured meeting information using Mistral AI
- Generate professional Word documents

## API Keys Required

### ElevenLabs API Key
- Get your API key from: https://elevenlabs.io/
- Used for high-quality video transcription
- Supports multiple video formats

### Mistral API Key
- Get your API key from: https://console.mistral.ai/
- Used for:
  - Combining transcripts with speaker names
  - Extracting structured meeting information
  - Processing historical documents (OCR)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys:
```bash
export ELEVENLABS_API_KEY="your_elevenlabs_key_here"
export MISTRAL_API_KEY="your_mistral_key_here"
```

3. Run the application:
```bash
streamlit run main.py
```

## Usage

### New Workflow (Recommended)
1. Select "Télécharger Vidéo + Teams"
2. Upload your meeting video
3. Paste the Teams transcript
4. Click "Traiter avec ElevenLabs + Mistral"
5. Review the combined transcript
6. Generate the meeting document

### Legacy Workflows
- **Whisper Audio**: Upload audio files for transcription
- **Manual Input**: Enter transcript manually

## Benefits of New Workflow

### ✅ High Accuracy
- ElevenLabs provides superior transcription quality
- Better handling of French language and technical terms

### ✅ Speaker Identification
- Combines accurate content with Teams speaker names
- Maintains chronological order and context

### ✅ Cost Effective
- Uses Mistral instead of Deepseek
- More efficient API usage

### ✅ Better Results
- Improved data extraction quality
- More reliable document generation

## File Structure

```
meeting-audio-transcript/
├── main.py                          # Main application
├── meeting_processor.py             # Historical document processor
├── process_historical_meetings.py   # Historical meeting processor
├── mistral.py                       # Legacy Mistral functions
├── requirements.txt                 # Dependencies
├── test_new_workflow.py            # Test script
├── processed_meetings/              # Historical meeting data
└── README_NEW_WORKFLOW.md          # This file
```

## Testing

Run the test script to verify your setup:
```bash
python test_new_workflow.py
```

## Troubleshooting

### ElevenLabs Issues
- Check API key validity
- Ensure video format is supported
- Verify file size limits

### Mistral Issues
- Verify API key and quota
- Check model availability
- Monitor token usage

### Common Problems
1. **API Key Errors**: Ensure both keys are set correctly
2. **File Upload Issues**: Check file format and size
3. **Transcription Failures**: Verify video/audio quality
4. **Memory Issues**: Large files may require more RAM

## Migration from Old Workflow

### What Changed
- Deepseek → Mistral for AI processing
- Whisper → ElevenLabs for transcription
- New combined workflow for better accuracy

### What Stayed the Same
- Document generation functionality
- Historical context processing
- Word document formatting
- Team member management

## Support

For issues or questions:
1. Check the troubleshooting section
2. Run the test script
3. Verify API keys and quotas
4. Check file formats and sizes 