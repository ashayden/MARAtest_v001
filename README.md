# Multi-Agent Gemini Chatbot

An interactive chatbot powered by Google's Gemini AI, featuring a multi-agent workflow for enhanced reasoning capabilities. Built with Streamlit for a seamless user experience.

## Features

- Interactive chat interface
- Multi-agent system for improved reasoning
- Powered by Google's Gemini AI
- Real-time responses
- Clean and intuitive UI

## Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

5. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── agents/               # Multi-agent system components
│   ├── __init__.py
│   ├── base_agent.py    # Base agent class
│   └── specialist.py    # Specialist agents
├── utils/               # Utility functions
│   ├── __init__.py
│   └── helpers.py
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Contributing

Feel free to open issues or submit pull requests for any improvements.

## License

MIT License 