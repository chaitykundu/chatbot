import json
import re
from typing import List, Dict, Any, Optional
import tiktoken

class ImprovedExerciseChunker:
    """
    Enhanced chunking system for educational exercise JSON data with better
    semantic preservation and more accurate token counting.
    """
    
    def __init__(self, max_tokens_per_chunk=1500, min_tokens_per_chunk=300, model="cl100k_base"):
        self.max_tokens = max_tokens_per_chunk
        self.min_tokens = min_tokens_per_chunk
        try:
            self.tokenizer = tiktoken.get_encoding(model)
        except:
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Accurate token counting using tiktoken"""
        if self.tokenizer:
            return len(self.tokenizer.encode(str(text)))
        else:
            # Fallback to improved estimation
            return len(str(text).split()) * 1.3  # Better than char/4
    
    def extract_semantic_text(self, obj: Any, context: str = "") -> str:
        """Extract text while preserving semantic context"""
        if isinstance(obj, str):
            return f"{context}: {obj}" if context else obj
        elif isinstance(obj, dict):
            texts = []
            for key, value in obj.items():
                if value is not None and key != 'svg':  # Skip SVG content
                    semantic_context = key.replace('_', ' ').title()
                    texts.append(self.extract_semantic_text(value, semantic_context))
            return " | ".join(filter(None, texts))
        elif isinstance(obj, list):
            return " | ".join([self.extract_semantic_text(item, context) for item in obj])
        else:
            return str(obj) if obj is not None else ""
    
    def create_exercise_metadata_chunk(self, exercise: Dict[str, Any]) -> Dict[str, Any]:
        """Create a metadata-focused chunk for exercise overview"""
        metadata = exercise['exercise_metadata']
        main_data = exercise['exercise_content']['main_data']
        
        # Create concise section summaries
        sections = exercise['exercise_content'].get('sections', [])
        section_previews = []
        
        for i, section in enumerate(sections):
            preview = {
                'section_number': section.get('section_number', i+1),
                'question_type': 'multiple_choice' if section.get('question_options') else 'open_ended',
                'has_hint': bool(section.get('hint')),
                'has_solution': bool(section.get('solution')),
                'content_types': self._identify_content_types(section)
            }
            section_previews.append(preview)
        
        chunk_text = f"""
        Exercise: {metadata['class']} - Lesson {metadata['lesson_number']} - Exercise {metadata['exercise_number']}
        Topic: {metadata['topic']}
        Type: {metadata['exercise_type']}
        Total Sections: {metadata['total_sections']}
        
        Main Content: {self.extract_semantic_text(main_data)}
        
        Sections Overview: {json.dumps(section_previews, ensure_ascii=False)}
        """.strip()
        
        return {
            'chunk_type': 'exercise_metadata',
            'exercise_id': f"{metadata['class']}_{metadata['lesson_number']}_{metadata['exercise_number']}",
            'metadata': metadata,
            'main_data': main_data,
            'section_count': len(sections),
            'sections_preview': section_previews,
            'searchable_text': chunk_text,
            'token_count': self.count_tokens(chunk_text)
        }
    
    def create_section_chunks(self, exercise: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create focused chunks for individual sections"""
        chunks = []
        metadata = exercise['exercise_metadata']
        main_data = exercise['exercise_content']['main_data']
        sections = exercise['exercise_content'].get('sections', [])
        
        for section in sections:
            # Build comprehensive section content
            section_parts = []
            
            # Add context from exercise
            section_parts.append(f"Exercise Context: {self.extract_semantic_text(main_data)}")
            
            # Add section-specific data if exists
            if section.get('section_data'):
                section_parts.append(f"Section Data: {self.extract_semantic_text(section['section_data'])}")
            
            # Add question
            if section.get('question'):
                section_parts.append(f"Question: {self.extract_semantic_text(section['question'])}")
            
            # Add options for multiple choice
            if section.get('question_options'):
                options_text = " | ".join([
                    f"Option {i+1}: {self.extract_semantic_text(opt)}" 
                    for i, opt in enumerate(section['question_options'])
                ])
                section_parts.append(f"Answer Options: {options_text}")
            
            # Add hint if available
            if section.get('hint'):
                section_parts.append(f"Hint: {self.extract_semantic_text(section['hint'])}")
            
            # Add solution
            if section.get('solution'):
                section_parts.append(f"Solution: {self.extract_semantic_text(section['solution'])}")
            
            # Add full solution if available
            if section.get('full_solution'):
                section_parts.append(f"Full Solution: {self.extract_semantic_text(section['full_solution'])}")
            
            chunk_text = "\n\n".join(section_parts)
            token_count = self.count_tokens(chunk_text)
            
            chunk = {
                'chunk_type': 'section_content',
                'exercise_id': f"{metadata['class']}_{metadata['lesson_number']}_{metadata['exercise_number']}",
                'section_id': section.get('section_id'),
                'section_number': section.get('section_number'),
                'exercise_metadata': metadata,
                'section_data': section,
                'searchable_text': chunk_text,
                'token_count': token_count,
                'content_flags': {
                    'has_svg': self._has_svg_content(section),
                    'has_table': self._has_table_content(section),
                    'has_multiple_choice': bool(section.get('question_options')),
                    'has_hint': bool(section.get('hint')),
                    'has_solution': bool(section.get('solution')),
                    'has_full_solution': bool(section.get('full_solution'))
                }
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def _identify_content_types(self, section: Dict) -> List[str]:
        """Identify types of content in a section"""
        types = []
        if self._has_svg_content(section):
            types.append('visual_diagram')
        if self._has_table_content(section):
            types.append('table_data')
        if section.get('question_options'):
            types.append('multiple_choice')
        if 'math' in str(section).lower() or '$' in str(section):
            types.append('mathematical')
        return types
    
    def _has_svg_content(self, section: Dict) -> bool:
        """Check if section contains SVG graphics"""
        content_str = str(section)
        return '<svg' in content_str.lower()
    
    def _has_table_content(self, section: Dict) -> bool:
        """Check if section contains table data"""
        def check_for_tables(obj):
            if isinstance(obj, dict):
                if 'table' in obj and obj['table'] is not None:
                    return True
                if 'headers' in obj or 'rows_data' in obj:
                    return True
                return any(check_for_tables(v) for v in obj.values())
            elif isinstance(obj, list):
                return any(check_for_tables(item) for item in obj)
            return False
        
        return check_for_tables(section)
    
    def chunk_exercises(self, exercises: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main chunking method with improved strategy:
        1. Create one metadata chunk per exercise
        2. Create focused content chunks per section
        3. Combine small sections only when semantically related
        """
        all_chunks = []
        
        for exercise in exercises:
            # Always create exercise metadata chunk
            metadata_chunk = self.create_exercise_metadata_chunk(exercise)
            all_chunks.append(metadata_chunk)
            
            # Create section content chunks
            section_chunks = self.create_section_chunks(exercise)
            
            # Handle small chunks by combining related ones
            processed_chunks = self._handle_small_chunks(section_chunks)
            all_chunks.extend(processed_chunks)
        
        return all_chunks
    
    def _handle_small_chunks(self, section_chunks: List[Dict]) -> List[Dict]:
        """Intelligently handle chunks that are too small"""
        if not section_chunks:
            return []
        
        processed = []
        current_combined = None
        
        for chunk in section_chunks:
            if chunk['token_count'] >= self.min_tokens:
                # Chunk is large enough
                if current_combined:
                    processed.append(current_combined)
                    current_combined = None
                processed.append(chunk)
            else:
                # Chunk is too small - combine with similar content
                if current_combined is None:
                    current_combined = self._start_combined_chunk(chunk)
                else:
                    current_combined = self._add_to_combined_chunk(current_combined, chunk)
                
                # If combined chunk is now large enough, finalize it
                if current_combined['token_count'] >= self.min_tokens:
                    processed.append(current_combined)
                    current_combined = None
        
        # Don't forget the last combined chunk
        if current_combined:
            processed.append(current_combined)
        
        return processed
    
    def _start_combined_chunk(self, first_chunk: Dict) -> Dict:
        """Initialize a combined chunk"""
        return {
            'chunk_type': 'combined_sections',
            'exercise_id': first_chunk['exercise_id'],
            'exercise_metadata': first_chunk['exercise_metadata'],
            'combined_sections': [first_chunk['section_data']],
            'searchable_text': first_chunk['searchable_text'],
            'token_count': first_chunk['token_count'],
            'content_flags': first_chunk['content_flags'].copy()
        }
    
    def _add_to_combined_chunk(self, combined_chunk: Dict, new_chunk: Dict) -> Dict:
        """Add a new chunk to an existing combined chunk"""
        combined_chunk['combined_sections'].append(new_chunk['section_data'])
        combined_chunk['searchable_text'] += f"\n\n--- Next Section ---\n\n{new_chunk['searchable_text']}"
        combined_chunk['token_count'] += new_chunk['token_count']
        
        # Merge content flags
        for flag, value in new_chunk['content_flags'].items():
            combined_chunk['content_flags'][flag] = combined_chunk['content_flags'].get(flag, False) or value
        
        return combined_chunk
    
    def analyze_chunking_results(self, chunks: List[Dict]) -> Dict:
        """Analyze the chunking results for optimization"""
        analysis = {
            'total_chunks': len(chunks),
            'chunk_types': {},
            'token_distribution': [],
            'content_flags_stats': {},
            'avg_tokens_per_chunk': 0
        }
        
        total_tokens = 0
        for chunk in chunks:
            # Count chunk types
            chunk_type = chunk['chunk_type']
            analysis['chunk_types'][chunk_type] = analysis['chunk_types'].get(chunk_type, 0) + 1
            
            # Token distribution
            token_count = chunk['token_count']
            analysis['token_distribution'].append(token_count)
            total_tokens += token_count
            
            # Content flags
            if 'content_flags' in chunk:
                for flag, value in chunk['content_flags'].items():
                    if flag not in analysis['content_flags_stats']:
                        analysis['content_flags_stats'][flag] = 0
                    if value:
                        analysis['content_flags_stats'][flag] += 1
        
        analysis['avg_tokens_per_chunk'] = total_tokens / len(chunks) if chunks else 0
        analysis['token_distribution'].sort()
        
        return analysis

# Usage example
def main():
    # Load the JSON data
    with open('Files/exercises_schema_v2_2025-09-22.json', 'r', encoding='utf-8') as f:
        exercises = json.load(f)
    
    # Initialize improved chunker
    chunker = ImprovedExerciseChunker(max_tokens_per_chunk=1500, min_tokens_per_chunk=300)
    
    # Create chunks
    chunks = chunker.chunk_exercises(exercises)
    
    # Analyze results
    analysis = chunker.analyze_chunking_results(chunks)
    
    print(f"Created {analysis['total_chunks']} chunks from {len(exercises)} exercises")
    print(f"Average tokens per chunk: {analysis['avg_tokens_per_chunk']:.1f}")
    print(f"Token distribution (min/max): {min(analysis['token_distribution'])}/{max(analysis['token_distribution'])}")
    
    print("\nChunk type distribution:")
    for chunk_type, count in analysis['chunk_types'].items():
        print(f"  {chunk_type}: {count}")
    
    print("\nContent flags distribution:")
    for flag, count in analysis['content_flags_stats'].items():
        print(f"  {flag}: {count}")
    
    # Save chunks for inspection
    with open('improved_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()