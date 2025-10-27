"""Custom tools for email processing crew."""

from crewai.tools.base_tool import BaseTool
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel, Field
import re
import json


class EmailParserTool(BaseTool):
    """Tool for parsing and extracting structured information from email text."""
    
    name: str = "Email Parser Tool"
    description: str = "Parses email text to extract structured information like headers, sender, recipients, subject, etc."

    class EmailParseInput(BaseModel):
        """Input schema for EmailParserTool."""
        email_text: str = Field(..., description="The raw email text to parse")

    args_schema: Type[BaseModel] = EmailParseInput

    def _run(self, email_text: str) -> str:
        """Parse email text and return structured information."""
        
        # Extract email headers and content
        result = {
            "sender": self._extract_sender(email_text),
            "recipients": self._extract_recipients(email_text),
            "subject": self._extract_subject(email_text),
            "body": self._extract_body(email_text),
            "headers": self._extract_headers(email_text),
            "links": self._extract_links(email_text),
            "attachments": self._extract_attachments(email_text),
            "dates": self._extract_dates(email_text)
        }
        
        return json.dumps(result, indent=2)
    
    def _extract_sender(self, text: str) -> Optional[str]:
        """Extract sender information."""
        sender_patterns = [
            r'From:\s*([^\n\r]+)',
            r'from:\s*([^\n\r]+)',
            r'Sender:\s*([^\n\r]+)'
        ]
        
        for pattern in sender_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_recipients(self, text: str) -> Dict[str, Optional[str]]:
        """Extract recipient information."""
        recipients = {
            "to": None,
            "cc": None,
            "bcc": None
        }
        
        patterns = {
            "to": r'To:\s*([^\n\r]+)',
            "cc": r'CC:\s*([^\n\r]+)',
            "bcc": r'BCC:\s*([^\n\r]+)'
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                recipients[field] = match.group(1).strip()
        
        return recipients
    
    def _extract_subject(self, text: str) -> Optional[str]:
        """Extract subject line."""
        subject_patterns = [
            r'Subject:\s*([^\n\r]+)',
            r'subject:\s*([^\n\r]+)'
        ]
        
        for pattern in subject_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_body(self, text: str) -> str:
        """Extract email body content."""
        # Remove headers and get the main content
        lines = text.split('\n')
        body_start = 0
        
        # Find where headers end (look for empty line)
        for i, line in enumerate(lines):
            if line.strip() == '' and i > 0:
                body_start = i + 1
                break
        
        return '\n'.join(lines[body_start:]).strip()
    
    def _extract_headers(self, text: str) -> Dict[str, str]:
        """Extract all email headers."""
        headers = {}
        lines = text.split('\n')
        
        for line in lines:
            if ':' in line and not line.startswith(' '):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    headers[key] = value
            elif not line.strip():  # Empty line indicates end of headers
                break
        
        return headers
    
    def _extract_links(self, text: str) -> list:
        """Extract URLs from email text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)
    
    def _extract_attachments(self, text: str) -> list:
        """Extract attachment information."""
        attachment_patterns = [
            r'attachment[s]?:\s*([^\n\r]+)',
            r'attached[s]?:\s*([^\n\r]+)',
            r'Content-Disposition:\s*attachment[^;]*;\s*filename=([^\n\r;]+)'
        ]
        
        attachments = []
        for pattern in attachment_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            attachments.extend(matches)
        
        return [att.strip('"\'') for att in attachments]
    
    def _extract_dates(self, text: str) -> list:
        """Extract date/time information."""
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return list(set(dates))  # Remove duplicates


class SpamIndicatorTool(BaseTool):
    """Tool for detecting spam indicators in email content."""
    
    name: str = "Spam Indicator Tool"
    description: str = "Analyzes email content for common spam indicators and suspicious patterns."

    class SpamAnalysisInput(BaseModel):
        """Input schema for SpamIndicatorTool."""
        email_text: str = Field(..., description="The email text to analyze for spam indicators")

    args_schema: Type[BaseModel] = SpamAnalysisInput

    def _run(self, email_text: str) -> str:
        """Analyze email for spam indicators."""
        
        indicators = {
            "urgency_words": self._check_urgency_words(email_text),
            "money_mentions": self._check_money_mentions(email_text),
            "suspicious_links": self._check_suspicious_links(email_text),
            "poor_grammar": self._check_grammar_issues(email_text),
            "caps_lock_abuse": self._check_caps_abuse(email_text),
            "phishing_phrases": self._check_phishing_phrases(email_text),
            "spam_score": 0
        }
        
        # Calculate spam score
        score = 0
        for key, value in indicators.items():
            if key != "spam_score" and value:
                if isinstance(value, list):
                    score += len(value) * 10
                else:
                    score += 20
        
        indicators["spam_score"] = min(score, 100)
        
        return json.dumps(indicators, indent=2)
    
    def _check_urgency_words(self, text: str) -> list:
        """Check for urgency-inducing words."""
        urgency_words = [
            "urgent", "immediately", "act now", "limited time", "expires",
            "hurry", "rush", "deadline", "final notice", "last chance",
            "claim now", "don't wait", "instant", "asap", "emergency"
        ]
        
        found_words = []
        text_lower = text.lower()
        for word in urgency_words:
            if word in text_lower:
                found_words.append(word)
        
        return found_words
    
    def _check_money_mentions(self, text: str) -> list:
        """Check for money-related spam indicators."""
        money_patterns = [
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            r'\d+(?:,\d{3})*(?:\.\d{2})?\s*dollars?',
            r'free\s+money',
            r'easy\s+money',
            r'make\s+money',
            r'earn\s+\$\d+',
            r'prize\s+money',
            r'cash\s+prize'
        ]
        
        found_money = []
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_money.extend(matches)
        
        return found_money
    
    def _check_suspicious_links(self, text: str) -> list:
        """Check for suspicious link patterns."""
        suspicious_patterns = [
            r'bit\.ly/\w+',
            r'tinyurl\.com/\w+',
            r'click\s+here',
            r'http://[^/]*[0-9]+[^/]*/',  # IP addresses
            r'https?://[^/]*\.tk/',       # Suspicious TLD
            r'https?://[^/]*\.ml/',       # Suspicious TLD
        ]
        
        suspicious_links = []
        for pattern in suspicious_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            suspicious_links.extend(matches)
        
        return suspicious_links
    
    def _check_grammar_issues(self, text: str) -> list:
        """Check for common grammar issues in spam emails."""
        grammar_issues = []
        
        # Multiple exclamation marks
        if re.search(r'!{2,}', text):
            grammar_issues.append("Multiple exclamation marks")
        
        # Multiple question marks
        if re.search(r'\?{2,}', text):
            grammar_issues.append("Multiple question marks")
        
        # Excessive capitalization
        if re.search(r'[A-Z]{5,}', text):
            grammar_issues.append("Excessive capitalization")
        
        # Common misspellings
        misspellings = ["recieve", "occured", "seperate", "definately"]
        for word in misspellings:
            if word in text.lower():
                grammar_issues.append(f"Misspelling: {word}")
        
        return grammar_issues
    
    def _check_caps_abuse(self, text: str) -> bool:
        """Check for excessive use of capital letters."""
        if len(text) == 0:
            return False
        
        caps_count = sum(1 for char in text if char.isupper())
        caps_ratio = caps_count / len(text)
        
        return caps_ratio > 0.3  # More than 30% caps
    
    def _check_phishing_phrases(self, text: str) -> list:
        """Check for common phishing phrases."""
        phishing_phrases = [
            "verify your account", "confirm your identity", "update your information",
            "suspended account", "unusual activity", "click to verify",
            "temporary hold", "security alert", "immediate action required",
            "congratulations you've won", "selected winner", "claim your prize"
        ]
        
        found_phrases = []
        text_lower = text.lower()
        for phrase in phishing_phrases:
            if phrase in text_lower:
                found_phrases.append(phrase)
        
        return found_phrases