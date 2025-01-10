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

```python
TAVILY_API_KEY='your_tavily_api_key'
DATAFORSEO_USERNAME='your_dataforseo_username'
DATAFORSEO_PASSWORD='your_dataforseo_password'
```

2. Install the required Python packages.

3. Run the script:

```python
python agent.py
```

---

## TODO

- [ ] Allow any type of site to be found. I.e. not just looking for social media marketing profesionals.
- [ ] Add a Google My Business search tool.
- [ ] Add a way to rank previous query searchs for in-prompt optimisation based on historical success.
- [ ] Make the websites that we decide to classify more specific.
- [ ] Build a browser agent to automatically find the contact information.
- [ ] Handle Captchas, rate limits and other errors.
