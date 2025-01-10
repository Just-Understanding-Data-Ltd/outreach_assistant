# tracker.py
import os
import json
from datetime import datetime
from urllib.parse import urlparse

def get_domain(url: str) -> str:
    """
    Extract domain from a URL, ignoring 'www.'.
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

class OutreachTracker:
    """
    Tracks used queries (google/tavily) and visited domains with classification data.
    """
    def __init__(self, filename="outreach_tracker.json"):
        self.filename = filename
        default_data = {
            "google_queries": [],
            "tavily_queries": [],
            "visited_domains": {}
        }

        if not os.path.exists(self.filename):
            with open(self.filename, "w", encoding="utf-8") as f:
                json.dump(default_data, f, indent=2)

        with open(self.filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = default_data

        self.google_queries = set(data.get("google_queries", []))
        self.tavily_queries = set(data.get("tavily_queries", []))
        self.visited_domains = data.get("visited_domains", {})

    def get_solo_count(self) -> int:
        return sum(1 for info in self.visited_domains.values() if info.get("classification") == "SOLO")


    def save(self):
        data = {
            "google_queries": list(self.google_queries),
            "tavily_queries": list(self.tavily_queries),
            "visited_domains": self.visited_domains
        }
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def has_google_query(self, query: str) -> bool:
        return (query in self.google_queries)

    def add_google_query(self, query: str):
        if not self.has_google_query(query):
            self.google_queries.add(query)
            self.save()

    def has_tavily_query(self, query: str) -> bool:
        return (query in self.tavily_queries)

    def add_tavily_query(self, query: str):
        if not self.has_tavily_query(query):
            self.tavily_queries.add(query)
            self.save()

    def has_visited_domain(self, url: str) -> bool:
        d = get_domain(url)
        return d in self.visited_domains

    def add_domain(self, url: str, classification: str, score: int, notes: str = "", screenshot_file: str = ""):
        """
        Store domain record with classification, numeric score, notes, etc.
        """
        d = get_domain(url)
        if d not in self.visited_domains:
            self.visited_domains[d] = {
                "first_visit_date": datetime.now().isoformat(),
                "classification": classification,
                "score": score,
                "notes": notes,
                "screenshot_file": screenshot_file,
            }
        else:
            rec = self.visited_domains[d]
            # If new classification is better than old? Up to you.
            if not rec["classification"]:
                rec["classification"] = classification
            if not rec["score"]:
                rec["score"] = score
            rec["notes"] += f"\n{notes}"
            if screenshot_file:
                rec["screenshot_file"] = screenshot_file

        self.save()

    def get_visited_count(self) -> int:
        return len(self.visited_domains)

    def get_all_domains(self):
        return self.visited_domains
