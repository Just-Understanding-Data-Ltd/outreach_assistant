import os
import time
import logging
import base64
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, Browser
from dataforseo_client import configuration as dfs_config, api_client as dfs_api_provider
from dataforseo_client.api.serp_api import SerpApi
from dataforseo_client.rest import ApiException
from dataforseo_client.models.serp_google_organic_live_advanced_request_info import SerpGoogleOrganicLiveAdvancedRequestInfo
from tavily import TavilyClient
from helpers import prune_messages_with_tool_pairs
from tracker import OutreachTracker, get_domain

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY", "")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in .env")

DATAFORSEO_USER = os.getenv("DATAFORSEO_USERNAME", "")
DATAFORSEO_PASS = os.getenv("DATAFORSEO_PASSWORD", "")
TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY", "")
TARGET_LEADS = 100
MAX_MESSAGE_COUNT = 30
CONSECUTIVE_ZERO_RESULTS_LIMIT = 3
BROWSER: Browser = None

if not DATAFORSEO_USER or not DATAFORSEO_PASS or not TAVILY_API_KEY:
    raise ValueError("Missing DATAFORSEO_USERNAME, DATAFORSEO_PASSWORD, or TAVILY_API_KEY in .env")

client = OpenAI(api_key=openai_api_key)
tracker = OutreachTracker()

def init_browser():
    global BROWSER
    if BROWSER is None:
        p = sync_playwright().start()
        BROWSER = p.chromium.launch(headless=True)
    return BROWSER

def close_browser():
    global BROWSER
    if BROWSER is not None:
        BROWSER.close()
        BROWSER = None

def search_dataforseo(query: str) -> list[str]:
    """
    Use DataForSEO to find relevant URLs for a given query.
    """
    logger.info(f"search_dataforseo('{query}') called.")
    if tracker.has_google_query(query):
        logger.info(f"Already used Google query: {query}")
        return []
    tracker.add_google_query(query)

    results = []
    configuration = dfs_config.Configuration(username=DATAFORSEO_USER, password=DATAFORSEO_PASS)
    with dfs_api_provider.ApiClient(configuration) as api_client:
        serp_api = SerpApi(api_client)
        try:
            api_response = serp_api.google_organic_live_advanced([
                SerpGoogleOrganicLiveAdvancedRequestInfo(
                    language_name="English",
                    location_name="United States",
                    keyword=query,
                    depth=50
                )
            ])
            tasks = api_response.tasks or []
            for t in tasks:
                for item in (t.result or []):
                    for o in getattr(item, "items", []):
                        if getattr(o, "type", None) == "organic":
                            if hasattr(o, "url"):
                                results.append(o.url)
        except ApiException as e:
            logger.error(f"DataForSEO error: {e}")
    logger.debug(f"search_dataforseo => {results}")
    return results

def search_tavily(query: str) -> list[str]:
    """
    Use Tavily API to find relevant URLs for a given query.
    """
    logger.info(f"search_tavily('{query}') called.")
    if tracker.has_tavily_query(query):
        logger.info(f"Already used Tavily query: {query}")
        return []
    tracker.add_tavily_query(query)

    results = []
    tclient = TavilyClient(api_key=TAVILY_API_KEY)
    try:
        resp = tclient.search(query)
        for r in resp.get("results", []):
            url = r.get("url")
            if not tracker.has_visited_domain(url):
                results.append(url)
    except Exception as e:
        logger.warning(f"Tavily error: {e}")

    logger.debug(f"search_tavily => {results}")
    return results

def do_screenshot_and_inspect(url: str) -> dict:
    """
    Takes a screenshot, calls GPT-4 Vision to classify SOLO/AGENCY/UNKNOWN + 1..10 score,
    then automatically saves the domain to outreach_tracker.json so that visited_domains is updated.

    (MODIFIED) We only navigate to the homepage now, e.g. 'https://{domain}'.
    """
    logger.info(f"do_screenshot_and_inspect({url}) invoked.")

    browser = init_browser()
    domain = get_domain(url)

    # If we already visited the domain, skip
    if tracker.has_visited_domain(url):
        return {
            "url": url,
            "classification": "UNKNOWN",
            "score": 1,
            "notes": "Domain previously visited.",
            "screenshot_file": ""
        }

    # Attempt to navigate + screenshot the homepage instead of the original URL
    homepage_url = f"https://{domain}"
    page = browser.new_page()
    screenshot_folder = "screenshots"
    if not os.path.exists(screenshot_folder):
        os.makedirs(screenshot_folder)
    screenshot_filename = os.path.join(
        screenshot_folder, f"screenshot_{int(time.time())}_{domain}.png"
    )
    try:
        # First try https
        page.goto(homepage_url, timeout=20000, wait_until="domcontentloaded")
    except Exception as e:
        logger.warning(f"HTTPS error for {domain}: {e}")
        # Then fallback to http if https fails
        try:
            page.goto(f"http://{domain}", timeout=20000, wait_until="domcontentloaded")
        except Exception as e2:
            logger.warning(f"HTTP error for {domain}: {e2}")
            page.close()
            return {
                "url": homepage_url,
                "classification": "ERROR",
                "score": 1,
                "notes": f"Navigation error: {e2}",
                "screenshot_file": ""
            }

    try:
        page.screenshot(path=screenshot_filename, full_page=True)
        page.close()
    except Exception as e:
        logger.warning(f"Playwright screenshot error: {e}")
        try:
            page.close()
        except:
            pass
        return {
            "url": homepage_url,
            "classification": "ERROR",
            "score": 1,
            "notes": f"Screenshot error: {e}",
            "screenshot_file": ""
        }

    # Convert screenshot to base64
    b64_img = ""
    try:
        with open(screenshot_filename, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Screenshot read error: {e}")
        return {
            "url": homepage_url,
            "classification": "ERROR",
            "score": 1,
            "notes": "Screenshot read error",
            "screenshot_file": screenshot_filename
        }



    class InspectOutput(BaseModel):
        url: str
        classification: Literal["SOLO","AGENCY","UNKNOWN"]
        score: Literal[1,2,3,4,5,6,7,8,9,10]
        notes: str

    vision_client = OpenAI(api_key=openai_api_key)
    system_content = (
        "You are GPT-4 with vision. You see a screenshot. "
        "Classify if this website is run by a SOLO freelancer or an AGENCY/aggregator. "
        "AGGREGATORS (like Upwork, Fiverr, Toptal, etc.) => classification='AGENCY' and score=1. "
        "A single person's personal site => classification='SOLO', score=10. If uncertain => classification='UNKNOWN' + mid-range score. "
        "Output must be valid JSON with keys: url, classification, score, notes. Strict schema."
    )

    prompt_messages = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": f"Classify site {homepage_url}. Return integer 'score' 1..10, plus 'classification' in [SOLO,AGENCY,UNKNOWN], plus 'notes'."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_img}",
                        "detail":"low"
                    }
                }
            ]
        }
    ]

    try:
        resp = vision_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=prompt_messages,
            response_format=InspectOutput
        )
        if not resp.choices:
            raise ValueError("No structured response from GPT-4 Vision")
        parsed = resp.choices[0].message.parsed
        if not parsed:
            return {
                "url": homepage_url,
                "classification": "UNKNOWN",
                "score": 1,
                "notes": "Vision classification parse error",
                "screenshot_file": screenshot_filename
            }

        # Convert pydantic object to dict
        final_result = {
            "url": parsed.url,
            "classification": parsed.classification,
            "score": parsed.score,
            "notes": parsed.notes,
            "screenshot_file": screenshot_filename
        }

        # (NEW) Automatically save the domain to outreach_tracker.json
        tracker.add_domain(
            url=final_result["url"],
            classification=final_result["classification"],
            score=final_result["score"],
            notes=final_result["notes"],
            screenshot_file=final_result["screenshot_file"]
        )

        return final_result

    except Exception as e:
        logger.error(f"Error parsing structured vision output: {e}")
        result = {
            "url": homepage_url,
            "classification": "ERROR",
            "score": 1,
            "notes": f"Vision classification parse error: {e}",
            "screenshot_file": screenshot_filename
        }
        # Add domain tracking even on error
        tracker.add_domain(
            url=result["url"],
            classification=result["classification"],
            score=result["score"],
            notes=result["notes"],
            screenshot_file=result["screenshot_file"]
        )
        return result

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_dataforseo",
            "description": "Perform a DataForSEO search for websites relevant to a given query.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the search query"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_tavily",
            "description": "Perform a Tavily search for websites relevant to a given query.",
            "strict": True,
            "parameters": {
                "type":"object",
                "properties":{
                    "query":{"type":"string","description":"the search query"}
                },
                "required":["query"],
                "additionalProperties":False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name":"screenshot_and_inspect",
            "description":"Screenshot + GPT-4 Vision => classification (SOLO/AGENCY/UNKNOWN), score(1..10), notes.",
            "strict":True,
            "parameters":{
                "type":"object",
                "properties":{
                    "url": {"type":"string"}
                },
                "required":["url"],
                "additionalProperties":False
            }
        }
    },
]




def run_agent_loop(target_leads=100, max_hours=4):
    f"""
    Example loop that tries to gather {target_leads} 'SOLO' leads. 
    'screenshot_and_inspect' automatically saves each domain as soon as it's classified.

    (MODIFIED) We now only stop once we've found `target_leads` SOLO websites.
    """
    start_time = time.time()
    deadline = start_time + max_hours*3600

    used_google = list(tracker.google_queries)
    used_tavily = list(tracker.tavily_queries)
    visited = list(tracker.visited_domains.keys())

    info_text = (
        f"Currently, google queries used: {used_google}\n"
        f"Tavily queries used: {used_tavily}\n"
        f"Visited domains so far: {visited[:20]}"
    )

    # Add zero results tracking
    zero_results_streak = 0
    previous_solo_count = tracker.get_solo_count()

    messages = [
        {
            "role": "system", 
            "content": (
                "You are an agent that can call:\n"
                "1) search_dataforseo(query)\n"
                "2) search_tavily(query)\n"
                "3) screenshot_and_inspect(url)\n"
                f"Goal: find at least {target_leads} SOLO websites.\n"
                "Be creative with searches - try different angles and synonyms.\n"
                "Avoid obvious agencies and aggregators like Upwork, Fiverr, etc.\n"
                "If multiple searches yield no results, pivot to new approaches. You must always search the URLs that seem promising with screenshot_and_inspect. "
                "You must stick to searching for social media marketing people. I really need to find social media marketing people, particularly"
            )
        },
        {
            "role":"assistant",
            "content":f"Here is what we know:\n{info_text}"
        },
        {
            "role":"user",
            "content":"Let's gather new leads for 'freelance social media marketing experts'."
        }
    ]

    while True:
        messages = prune_messages_with_tool_pairs(messages, max_count=MAX_MESSAGE_COUNT)
    
        # (MODIFIED) we only check how many SOLO leads we have
        solo_count = tracker.get_solo_count()

        if time.time() >= deadline:
            logger.info("Time limit reached. Stopping.")
            break

        if solo_count >= target_leads:
            logger.info(f"Reached {solo_count}/{target_leads} SOLO leads, done.")
            break

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            temperature=0.7,  # Increased for more variety
            parallel_tool_calls=True,
        )

        # Track if we found new leads
        current_solo_count = tracker.get_solo_count()
        if current_solo_count > previous_solo_count:
            zero_results_streak = 0
            previous_solo_count = current_solo_count
        else:
            zero_results_streak += 1

        # Force creative pivot after N consecutive misses
        if zero_results_streak >= CONSECUTIVE_ZERO_RESULTS_LIMIT:
            messages.append({
                "role": "user",
                "content": (
                    "We've had several queries with no new leads. Let's try a new approach!\n"
                    "1. Use completely different search terms like 'freelance social media marketing experts'. Get creative with google search operators.\n"
                    "2. Look for personal portfolio sites or blogs in marketing\n"
                    f"We still need {target_leads - current_solo_count} more SOLO leads."
                )
            })
            zero_results_streak = 0

        reply = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if not reply.tool_calls:
            logger.info(f"LLM says: {reply.content or ''}")
            if finish_reason in ("stop","length","content_filter"):
                if solo_count < target_leads:
                    # push back
                    logger.info("Model tried to finalize early, pushing it to continue.")
                    messages.append(reply)
                    messages.append({
                        "role":"user",
                        "content":f"We only have {solo_count} SOLO leads. We need {target_leads}!"
                    })
                    continue
                else:
                    break
            messages.append(reply)
            continue

        messages.append(reply)
        t_calls = reply.tool_calls
        tool_responses = []

        for tc in t_calls:
            call_id = tc.id
            fn_name = tc.function.name
            raw_args = tc.function.arguments
            logger.debug(f"Tool call => name={fn_name}, call_id={call_id}, arguments={raw_args}")

            import json
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
                continue

            if fn_name=="search_dataforseo":
                out = search_dataforseo(**parsed_args)
                tool_responses.append({
                    "role":"tool","name":fn_name,"tool_call_id":call_id,
                    "content":json.dumps(out)
                })
            elif fn_name=="search_tavily":
                out = search_tavily(**parsed_args)
                tool_responses.append({
                    "role":"tool","name":fn_name,"tool_call_id":call_id,
                    "content":json.dumps(out)
                })
            elif fn_name=="screenshot_and_inspect":
                out = do_screenshot_and_inspect(**parsed_args)
                tool_responses.append({
                    "role":"tool","name":fn_name,"tool_call_id":call_id,
                    "content":json.dumps(out)
                })
            else:
                logger.warning(f"Unknown function: {fn_name}")

        messages.extend(tool_responses)
        if finish_reason in ("stop","length","content_filter"):
            if solo_count < target_leads:
                logger.info("Model tried to finalize early.")
                messages.append({
                    "role":"user",
                    "content":f"Only {solo_count} SOLO leads so far. We must get {target_leads}. Continue searching!"
                })
                continue
            else:
                break

    close_browser()
    logger.info("Agent done.")
    total_solo = tracker.get_solo_count()
    logger.info(f"SOLO leads found: {total_solo}")
    for i, (dom, info) in enumerate(tracker.visited_domains.items(), start=1):
        logger.info(
            f"{i}. {dom} => {info['classification']} "
            f"(score={info.get('score','?')}) - {info['notes'][:60]}..."
        )


if __name__ == "__main__":
    run_agent_loop(target_leads=TARGET_LEADS, max_hours=4)