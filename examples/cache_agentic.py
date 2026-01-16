from typing import List, Dict, Optional
from dataclasses import dataclass
import re
from collections import Counter
import json

@dataclass
class TaggingConfig:
    max_tags: int = 3
    min_confidence: float = 0.7
    use_cache: bool = True
    max_tokens_per_request: int = 100

class MinimalPromptManager:
    """Managers prompts with token efficiency"""

    def __init__(self):
        self.cache = {}
        self.system_prompt = "Tag text. Output: JSON with tags and confidence"

    def get_tagging_prompt(self, text:str) -> str:
        """Ultra-minimal prompt construction"""

        # Cache prompts for similar texts
        text_key = text[:50]  # Use beginning as cache key

        if text_key in self.cache:
            return f"Tag like before: {text[:100]}"

        return f"{self.system_prompt}\nText: {text[:200]}"


class AutoTaggingAgent:
    """Efficient tagging agent with minimal API calls"""

    def __init__(self, config: Optional[TaggingConfig] = None):
        self.config = config or TaggingConfig()
        self.prompt_manager = MinimalPromptManager()
        self.tag_cache = {}

        self.common_tags = {
            'technology': ['ai', 'machine learning', 'programming', 'software'],
            'business': ['startup', 'finance', 'marketing', 'management'],
            'science': ['research', 'data', 'analysis', 'experiment'],
            'general': ['news', 'update', 'guide', 'tutorial']
        }

    def extract_tags(self, text: str) -> List[Dict[str, float]]:

        local_tags = self._extract_local_tags(text)
        if len(local_tags) >= self.config.max_tags:
            return local_tags[:self.config.max_tags]

        
        ai_tags = self._extract_ai_tags(text, existing_tags=local_tags)
        all_tags = self._merge_tags(local_tags + ai_tags)

        return all_tags[:self.config.max_tags]

    
    def _extract_local_tags(self, text: str) -> List[Dict[str, float]]:
            """Extract tags using local rules (no API calls)"""
            tags = []
            text_lower = text.lower()
            
            # Rule-based extraction
            for category, keywords in self.common_tags.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        # Simple confidence based on frequency and position
                        confidence = min(0.9, 0.3 + text_lower.count(keyword) * 0.1)
                        if confidence >= self.config.min_confidence:
                            tags.append({"tag": keyword, "confidence": round(confidence, 2)})
            
            # Extract hashtags if present
            hashtags = re.findall(r'#(\w+)', text)
            for tag in hashtags[:3]:  # Limit hashtags
                tags.append({"tag": tag.lower(), "confidence": 0.85})
            
            return sorted(tags, key=lambda x: x["confidence"], reverse=True)
        
    def _extract_ai_tags(self, text: str, existing_tags: List[Dict]) -> List[Dict]:
        """
        Use AI with minimal prompt for remaining tags
        Simulating GPT-5 Nano call with minimal token usage
        """
        # Construct minimal prompt
        prompt = self.prompt_manager.get_tagging_prompt(text)
        
        # Simulated AI response (in real use, call API with minimal tokens)
        # Format: JSON only, no explanations
        existing_tag_names = [t["tag"] for t in existing_tags]
        remaining_slots = self.config.max_tags - len(existing_tags)
        
        if remaining_slots <= 0:
            return []
        
        # Simulated efficient AI call
        # In reality, you'd call: response = openai_minimal_call(prompt)
        simulated_response = {
            "tags": [
                {"tag": "artificial intelligence", "confidence": 0.92},
                {"tag": "automation", "confidence": 0.88}
            ]
        }
        
        return simulated_response["tags"][:remaining_slots]
    
    def _merge_tags(self, tags: List[Dict]) -> List[Dict]:
        """Merge similar tags and deduplicate"""
        merged = {}
        for tag_info in tags:
            tag_name = tag_info["tag"]
            confidence = tag_info["confidence"]
            
            if tag_name not in merged or confidence > merged[tag_name]:
                merged[tag_name] = confidence
        
        # Convert back to list and sort
        result = [{"tag": k, "confidence": v} for k, v in merged.items()]
        return sorted(result, key=lambda x: x["confidence"], reverse=True)
    
    def batch_tag(self, texts: List[str]) -> List[List[Dict]]:
        """Process multiple texts efficiently"""
        return [self.extract_tags(text) for text in texts]
    
    def get_tag_summary(self, tags: List[Dict]) -> Dict:
        """Create summary statistics"""
        tag_names = [t["tag"] for t in tags]
        return {
            "total_tags": len(tags),
            "top_tag": tags[0]["tag"] if tags else None,
            "avg_confidence": sum(t["confidence"] for t in tags) / len(tags) if tags else 0,
            "tag_frequency": dict(Counter(tag_names))
        }


    
# Usage Example
def demonstrate_agent():
    """Showcase the agent's capabilities"""
    
    # Initialize agent with efficient configuration
    config = TaggingConfig(
        max_tags=5,
        min_confidence=0.6,
        max_tokens_per_request=150
    )
    
    agent = AutoTaggingAgent(config)
    
    # Sample texts
    sample_texts = [
        "AI and machine learning are revolutionizing healthcare with new diagnostic tools. #AI #HealthTech",
        "Startup funding reached new heights in Q3 with blockchain companies leading the charge.",
        "Climate change research shows alarming trends in polar ice melt rates."
    ]
    
    print("Auto-Tagging Agent Demonstration")
    print("=" * 50)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nText {i}: {text[:100]}...")
        tags = agent.extract_tags(text)
        summary = agent.get_tag_summary(tags)
        
        print(f"Tags extracted: {len(tags)}")
        for tag in tags:
            print(f"  - {tag['tag']} ({tag['confidence']})")
        
        print(f"Summary: {summary}")
        

if __name__ == "__main__":
    demonstrate_agent()