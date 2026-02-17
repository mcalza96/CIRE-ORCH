
import re
from dataclasses import dataclass

@dataclass
class QuerySplitterConfig:
    min_length: int = 10
    max_parts: int = 3
    enabled: bool = True

class QuerySplitter:
    def __init__(self, config: QuerySplitterConfig | None = None):
        self.config = config or QuerySplitterConfig()

    def split(self, query: str) -> list[str]:
        if not self.config.enabled:
            return [query]
        
        raw = str(query or "").strip()
        if not raw:
            return []
            
        # Split by question marks or newlines
        parts = re.split(r"\?+|\n+", raw)
        out: list[str] = []
        seen: set[str] = set()
        
        for part in parts:
            candidate = " ".join(part.split()).strip(" .:-")
            if len(candidate) < self.config.min_length:
                continue
            
            key = candidate.lower()
            if key in seen:
                continue
                
            seen.add(key)
            out.append(candidate)
            
            if len(out) >= self.config.max_parts:
                break
                
        # Only return parts if we found at least 2 valid ones, otherwise return original query as single item
        return out if len(out) >= 2 else [query]
