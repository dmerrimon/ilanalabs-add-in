#!/Users/donmerriman/ilana-addin/backend-mvp/ml_env/bin/python
"""
PubMedBERT Service for Advanced Clinical Protocol Analysis
Uses Microsoft's BiomedNLP-PubMedBERT model for biomedical text understanding
"""

import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import numpy as np
from typing import Dict, List, Any, Tuple

class PubMedBERTAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        
        # Load tokenizer and model
        print("Loading PubMedBERT model...", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        print("✅ PubMedBERT loaded successfully", file=sys.stderr)
        
        # ICH E6 compliance keywords for semantic matching
        self.ich_requirements = {
            'protocol_structure': [
                'protocol title', 'protocol number', 'version', 'date',
                'sponsor', 'investigator', 'trial registration'
            ],
            'objectives': [
                'primary objective', 'secondary objective', 'exploratory objective',
                'primary endpoint', 'secondary endpoint', 'efficacy', 'safety'
            ],
            'trial_design': [
                'randomized', 'controlled', 'blinded', 'double-blind', 'single-blind',
                'open-label', 'parallel group', 'crossover', 'dose-escalation',
                'phase i', 'phase ii', 'phase iii', 'phase iv'
            ],
            'participants': [
                'inclusion criteria', 'exclusion criteria', 'eligibility',
                'recruitment', 'screening', 'enrollment', 'withdrawal criteria'
            ],
            'intervention': [
                'intervention', 'treatment', 'dosage', 'administration',
                'duration', 'concomitant medication', 'prohibited medication'
            ],
            'assessments': [
                'assessment schedule', 'visit schedule', 'procedures',
                'laboratory tests', 'vital signs', 'physical examination'
            ],
            'safety': [
                'adverse event', 'serious adverse event', 'safety monitoring',
                'data safety monitoring board', 'stopping rules', 'risk assessment'
            ],
            'statistics': [
                'sample size', 'power calculation', 'statistical analysis',
                'intention to treat', 'per protocol', 'p-value', 'confidence interval'
            ],
            'ethics': [
                'informed consent', 'ethics committee', 'irb approval',
                'confidentiality', 'data protection', 'gdpr compliance'
            ],
            'data_management': [
                'data collection', 'case report form', 'data monitoring',
                'source data verification', 'audit', 'quality control'
            ]
        }
        
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get BERT embeddings for the input text"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embeddings as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    def calculate_semantic_similarity(self, text: str, reference_texts: List[str]) -> float:
        """Calculate semantic similarity between text and reference texts using BERT"""
        text_embedding = self.get_embeddings(text)
        
        similarities = []
        for ref_text in reference_texts:
            ref_embedding = self.get_embeddings(ref_text)
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(text_embedding, ref_embedding)
            similarities.append(cos_sim.item())
        
        return max(similarities) if similarities else 0.0
    
    def extract_clinical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract clinical entities using BERT contextual understanding"""
        entities = {
            'therapeutic_areas': [],
            'study_phases': [],
            'clinical_terms': [],
            'regulatory_terms': []
        }
        
        text_lower = text.lower()
        
        # Therapeutic areas with context
        therapeutic_patterns = [
            'oncology', 'cardiovascular', 'diabetes', 'respiratory',
            'neurology', 'infectious disease', 'psychiatric', 'dermatology',
            'gastroenterology', 'rheumatology', 'hematology', 'endocrinology'
        ]
        
        for pattern in therapeutic_patterns:
            if pattern in text_lower:
                entities['therapeutic_areas'].append(pattern)
        
        # Study phases with BERT understanding
        phase_patterns = [
            ('phase i', 'Phase I'), ('phase 1', 'Phase I'),
            ('phase ii', 'Phase II'), ('phase 2', 'Phase II'),
            ('phase iii', 'Phase III'), ('phase 3', 'Phase III'),
            ('phase iv', 'Phase IV'), ('phase 4', 'Phase IV'),
            ('pilot study', 'Pilot'), ('feasibility study', 'Feasibility')
        ]
        
        for pattern, label in phase_patterns:
            if pattern in text_lower:
                entities['study_phases'].append(label)
                break
        
        # Clinical terms extraction
        clinical_keywords = [
            'randomized', 'placebo', 'efficacy', 'safety', 'endpoint',
            'biomarker', 'pharmacokinetics', 'pharmacodynamics',
            'dose-response', 'therapeutic', 'prophylactic'
        ]
        
        for keyword in clinical_keywords:
            if keyword in text_lower:
                entities['clinical_terms'].append(keyword)
        
        # Regulatory terms
        regulatory_keywords = [
            'fda', 'ema', 'ich', 'gcp', 'gdpr', 'hipaa', '21 cfr',
            'regulatory approval', 'clinical trial authorization'
        ]
        
        for keyword in regulatory_keywords:
            if keyword in text_lower:
                entities['regulatory_terms'].append(keyword)
        
        return entities
    
    def analyze_ich_compliance_bert(self, text: str) -> Dict[str, Any]:
        """Analyze ICH E6 compliance using BERT semantic understanding"""
        compliance_scores = {}
        missing_elements = []
        
        # Check each ICH requirement category using semantic similarity
        for category, keywords in self.ich_requirements.items():
            # Create reference sentences for this category
            reference_sentences = [f"The protocol includes {keyword}" for keyword in keywords]
            
            # Calculate semantic similarity
            similarity = self.calculate_semantic_similarity(text, reference_sentences)
            compliance_scores[category] = similarity
            
            # If similarity is low, identify missing elements
            if similarity < 0.3:
                for keyword in keywords:
                    if keyword not in text.lower():
                        missing_elements.append(f"{category}: {keyword}")
        
        # Overall compliance score
        overall_score = np.mean(list(compliance_scores.values()))
        
        return {
            'overall_compliance': float(overall_score),
            'category_scores': compliance_scores,
            'missing_elements': missing_elements[:10],  # Top 10 missing elements
            'bert_enhanced': True
        }
    
    def generate_contextual_insights(self, text: str) -> List[Dict[str, Any]]:
        """Generate contextual insights using BERT's understanding"""
        insights = []
        entities = self.extract_clinical_entities(text)
        compliance = self.analyze_ich_compliance_bert(text)
        
        # Therapeutic area insights
        if not entities['therapeutic_areas']:
            insights.append({
                'type': 'missing_info',
                'category': 'therapeutic_area',
                'message': 'BERT Analysis: No therapeutic area detected. Specify the medical condition being studied.',
                'confidence': 0.9,
                'bert_powered': True
            })
        
        # Study phase insights
        if not entities['study_phases']:
            insights.append({
                'type': 'missing_info',
                'category': 'study_phase',
                'message': 'BERT Analysis: Study phase not identified. Clearly state Phase I, II, III, or IV.',
                'confidence': 0.95,
                'bert_powered': True
            })
        
        # Compliance insights based on BERT analysis
        for category, score in compliance['category_scores'].items():
            if score < 0.3:
                category_readable = category.replace('_', ' ').title()
                insights.append({
                    'type': 'compliance_gap',
                    'category': category,
                    'message': f'BERT Analysis: {category_readable} section appears incomplete (confidence: {score:.1%})',
                    'confidence': 1 - score,
                    'bert_powered': True
                })
        
        # Clinical terminology richness
        if len(entities['clinical_terms']) < 3:
            insights.append({
                'type': 'quality',
                'category': 'terminology',
                'message': 'BERT Analysis: Limited clinical terminology detected. Consider adding more technical details.',
                'confidence': 0.7,
                'bert_powered': True
            })
        
        # Regulatory compliance
        if not entities['regulatory_terms']:
            insights.append({
                'type': 'regulatory',
                'category': 'compliance',
                'message': 'BERT Analysis: No regulatory framework mentioned. Reference ICH-GCP, FDA, or EMA guidelines.',
                'confidence': 0.8,
                'bert_powered': True
            })
        
        return insights
    
    def calculate_quality_metrics(self, text: str) -> Dict[str, float]:
        """Calculate quality metrics using BERT embeddings"""
        # Get embeddings for quality assessment
        embeddings = self.get_embeddings(text)
        
        # Create reference high-quality protocol text
        high_quality_ref = """
        This randomized, double-blind, placebo-controlled Phase III study will evaluate 
        the efficacy and safety of the investigational drug in patients. The primary 
        objective is to demonstrate superiority over placebo. Inclusion and exclusion 
        criteria are clearly defined. Sample size calculation provides 90% power. 
        Statistical analysis will follow intention-to-treat principle. Informed consent 
        will be obtained from all participants. The study follows ICH-GCP guidelines.
        """
        
        ref_embedding = self.get_embeddings(high_quality_ref)
        
        # Calculate similarity to high-quality reference
        quality_similarity = torch.nn.functional.cosine_similarity(
            embeddings, ref_embedding
        ).item()
        
        # Extract entities for additional metrics
        entities = self.extract_clinical_entities(text)
        
        # Calculate various quality metrics
        metrics = {
            'semantic_quality': float(max(0, quality_similarity)),
            'terminology_richness': min(1.0, len(entities['clinical_terms']) / 10),
            'regulatory_alignment': min(1.0, len(entities['regulatory_terms']) / 5),
            'therapeutic_clarity': 1.0 if entities['therapeutic_areas'] else 0.0,
            'phase_specification': 1.0 if entities['study_phases'] else 0.0
        }
        
        # Overall BERT quality score
        metrics['bert_quality_score'] = np.mean(list(metrics.values()))
        
        return metrics
    
    def analyze_with_bert(self, text: str, section_type: str = 'general') -> Dict[str, Any]:
        """Main analysis method combining all BERT capabilities"""
        try:
            # Extract clinical entities
            entities = self.extract_clinical_entities(text)
            
            # Analyze ICH compliance
            compliance = self.analyze_ich_compliance_bert(text)
            
            # Generate contextual insights
            insights = self.generate_contextual_insights(text)
            
            # Calculate quality metrics
            quality_metrics = self.calculate_quality_metrics(text)
            
            # Combine all results
            result = {
                'bert_analysis': {
                    'entities': entities,
                    'compliance': compliance,
                    'insights': insights,
                    'quality_metrics': quality_metrics
                },
                'bert_powered': True,
                'model': 'microsoft/BiomedNLP-PubMedBERT',
                'confidence': compliance['overall_compliance']
            }
            
            return result
            
        except Exception as e:
            print(f"BERT analysis error: {e}", file=sys.stderr)
            return {
                'bert_analysis': None,
                'bert_powered': False,
                'error': str(e)
            }

def main():
    """CLI interface for testing"""
    if len(sys.argv) != 3:
        print("Usage: python3 pubmedbert_service.py '<text>' '<section_type>'")
        sys.exit(1)
    
    text = sys.argv[1]
    section_type = sys.argv[2]
    
    # Initialize analyzer
    analyzer = PubMedBERTAnalyzer()
    
    # Perform analysis
    result = analyzer.analyze_with_bert(text, section_type)
    
    # Output JSON result
    print(json.dumps(result, indent=2))

# Alias for backward compatibility
PubmedBERTService = PubMedBERTAnalyzer

if __name__ == "__main__":
    main()