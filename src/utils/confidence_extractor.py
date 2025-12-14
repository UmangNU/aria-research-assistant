"""
Confidence Score Extraction
Extracts AI's confidence in its statements

RESPONSIBLE AI - Acknowledges uncertainty!
"""

from typing import Dict, Tuple
import re

class ConfidenceExtractor:
    """Extract and parse confidence scores from LLM outputs"""
    
    def __init__(self):
        self.name = "Confidence Extractor"
    
    def add_confidence_request(self, prompt: str) -> str:
        """
        Modify prompt to request confidence scoring
        
        Args:
            prompt: Original prompt
        
        Returns:
            Enhanced prompt requesting confidence
        """
        
        confidence_instruction = """

IMPORTANT: After your response, rate your confidence on a new line:
- Format: "Confidence: XX/100" where XX is 0-100
- 90-100: Very confident (well-established facts, strong evidence)
- 70-89: Confident (supported by multiple sources)
- 50-69: Moderate (some uncertainty, limited evidence)
- Below 50: Low confidence (speculative, needs more research)

"""
        
        return prompt + confidence_instruction
    
    def extract_confidence(self, response: str) -> Tuple[str, float, bool]:
        """
        Extract confidence score from response
        
        Args:
            response: LLM response
        
        Returns:
            (cleaned_text, confidence_score, has_confidence)
        """
        
        # Pattern 1: "Confidence: XX/100"
        pattern1 = r'Confidence:\s*(\d+)/100'
        match = re.search(pattern1, response, re.IGNORECASE)
        
        if match:
            confidence_value = int(match.group(1))
            confidence_normalized = confidence_value / 100.0
            
            # Remove confidence line from text
            cleaned_text = re.sub(pattern1, '', response, flags=re.IGNORECASE).strip()
            
            return cleaned_text, confidence_normalized, True
        
        # Pattern 2: "Confidence: XX"
        pattern2 = r'Confidence:\s*(\d+)'
        match = re.search(pattern2, response, re.IGNORECASE)
        
        if match:
            confidence_value = int(match.group(1))
            # Assume it's already 0-100 scale
            if confidence_value <= 1:
                confidence_normalized = confidence_value
            else:
                confidence_normalized = confidence_value / 100.0
            
            cleaned_text = re.sub(pattern2, '', response, flags=re.IGNORECASE).strip()
            
            return cleaned_text, confidence_normalized, True
        
        # No confidence found
        return response, 0.75, False  # Default: 75% confidence
    
    def format_confidence_badge(self, confidence: float) -> str:
        """Format confidence as display badge"""
        
        if confidence >= 0.9:
            return f"🟢 Very High ({confidence*100:.0f}%)"
        elif confidence >= 0.7:
            return f"🟡 High ({confidence*100:.0f}%)"
        elif confidence >= 0.5:
            return f"🟠 Moderate ({confidence*100:.0f}%)"
        else:
            return f"🔴 Low ({confidence*100:.0f}%)"
    
    def interpret_confidence(self, confidence: float) -> str:
        """Provide interpretation of confidence level"""
        
        if confidence >= 0.9:
            return "AI is very confident - well-established facts with strong evidence"
        elif confidence >= 0.7:
            return "AI is confident - supported by research literature"
        elif confidence >= 0.5:
            return "AI has moderate confidence - some uncertainty exists"
        else:
            return "AI has low confidence - speculative or limited evidence"