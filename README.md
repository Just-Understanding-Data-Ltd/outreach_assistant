# AI-Powered Outreach Assistant

An intelligent web research automation tool that uses AI to find and classify potential leads, specifically targeting solo freelancers and independent contractors in social media marketing.

## Features

- **Multi-Source Search**: Utilizes both DataForSEO and Tavily APIs for comprehensive web searches
- **AI-Powered Classification**: Uses GPT-4 Vision to analyze and classify websites
- **Automated Screenshot Capture**: Takes full-page screenshots of discovered websites
- **Smart Query Persistence**: Tracks and avoids duplicate searches
- **Domain Tracking**: Maintains a record of visited domains and their classifications
- **Intelligent Pivoting**: Automatically adjusts search strategies when results are suboptimal

## Prerequisites

- Python 3.8+
- OpenAI API key
- DataForSEO credentials
- Tavily API key
- Playwright

## Setup

1. Create a `.env` file in the root directory:

```
TAVILY_API_KEY='your_tavily_api_key'
DATAFORSEO_USERNAME='your_dataforseo_username'
DATAFORSEO_PASSWORD='your_dataforseo_password'
```

2. Install the required Python packages.

3. Run the script:

```
python agent.py
```
