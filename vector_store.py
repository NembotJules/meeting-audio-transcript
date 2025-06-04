from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os

class MeetingVectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the vector store with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.meetings: List[Dict] = []
        
    def add_meeting(self, meeting_data: Dict) -> None:
        """
        Add a meeting to the vector store.
        
        Args:
            meeting_data: Dictionary containing meeting information with keys:
                - transcript: str
                - date: str (DD/MM/YYYY)
                - title: str
                - extracted_info: Dict
        """
        # Create a comprehensive text representation of the meeting
        meeting_text = f"""
        Title: {meeting_data['title']}
        Date: {meeting_data['date']}
        
        Transcript:
        {meeting_data['transcript']}
        
        Extracted Information:
        {json.dumps(meeting_data['extracted_info'], indent=2)}
        """
        
        # Get embedding
        embedding = self.model.encode([meeting_text])[0]
        
        # Add to FAISS index
        self.index.add(np.array([embedding]).astype('float32'))
        
        # Store meeting data
        self.meetings.append(meeting_data)
    
    def get_relevant_context(self, query_text: str, k: int = 3) -> List[Dict]:
        """
        Retrieve k most relevant meetings for the given query.
        
        Args:
            query_text: Text to find relevant meetings for
            k: Number of meetings to retrieve
            
        Returns:
            List of relevant meeting dictionaries
        """
        # Get query embedding
        query_embedding = self.model.encode([query_text])[0]
        
        # Search in FAISS
        k = min(k, len(self.meetings))  # Don't try to retrieve more than we have
        if k == 0:
            return []
            
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), 
            k
        )
        
        # Return relevant meetings
        return [self.meetings[idx] for idx in indices[0]]
    
    def save(self, directory: str) -> None:
        """Save the vector store to disk."""
        os.makedirs(directory, exist_ok=True)
        
        # Save the FAISS index
        faiss.write_index(self.index, os.path.join(directory, "faiss.index"))
        
        # Save the meetings data
        with open(os.path.join(directory, "meetings.json"), "w") as f:
            json.dump(self.meetings, f)
    
    def load(self, directory: str) -> None:
        """Load the vector store from disk."""
        # Load the FAISS index
        self.index = faiss.read_index(os.path.join(directory, "faiss.index"))
        
        # Load the meetings data
        with open(os.path.join(directory, "meetings.json"), "r") as f:
            self.meetings = json.load(f) 