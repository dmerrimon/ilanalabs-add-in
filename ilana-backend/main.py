from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ilana Protocol Intelligence API",
    description="AI-powered clinical protocol analysis service",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://agreeable-forest-0bbaa4e0f.3.azurestaticapps.net",
        "https://agreeable-forest-0bbaa4e0f-preview.eastus2.3.azurestaticapps.net",
        "http://localhost:3000",
        "https://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Request models
class AnalysisOptions(BaseModel):
    analyze_compliance: bool = True
    analyze_clarity: bool = True
    analyze_engagement: bool = True
    analyze_delivery: bool = True
    analyze_safety: bool = True
    analyze_regulatory: bool = True
    comprehensive_mode: bool = True
    min_issues: int = 5

class ProtocolAnalysisRequest(BaseModel):
    text: str
    options: Optional[AnalysisOptions] = AnalysisOptions()

class Issue(BaseModel):
    type: str
    message: str
    suggestion: Optional[str] = None

class AnalysisResponse(BaseModel):
    compliance_score: int
    clarity_score: int
    engagement_score: int
    delivery_score: int
    issues: List[Issue]
    metadata: Dict[str, Any]

@app.post("/analyze-protocol", response_model=AnalysisResponse)
async def analyze_protocol(request: ProtocolAnalysisRequest):
    try:
        logger.info(f"Received analysis request for text length: {len(request.text)}")
        logger.info(f"Analysis options: {request.options}")
        
        # Here you would integrate with PubmedBERT and Azure OpenAI
        # This is a simplified example
        
        # Simulate comprehensive analysis
        scores = {
            "compliance_score": 85,
            "clarity_score": 78,
            "engagement_score": 82,
            "delivery_score": 79
        }
        
        # Generate comprehensive issues
        issues = [
            Issue(
                type="compliance",
                message="Patient eligibility criteria need more specific inclusion/exclusion parameters.",
                suggestion="Add measurable clinical parameters and laboratory value ranges."
            ),
            Issue(
                type="clarity",
                message="Visit scheduling instructions could be more explicit about timing windows.",
                suggestion="Specify acceptable visit windows (e.g., ±3 days) for each timepoint."
            ),
            Issue(
                type="safety",
                message="Adverse event reporting timeline needs clarification.",
                suggestion="Specify exact timeframes for different severity levels of AEs."
            ),
            Issue(
                type="engagement",
                message="Patient education materials reference is incomplete.",
                suggestion="Include specific patient-facing materials and delivery methods."
            ),
            Issue(
                type="delivery",
                message="Site training requirements are not sufficiently detailed.",
                suggestion="Add specific training modules and competency assessment criteria."
            ),
            Issue(
                type="regulatory",
                message="IRB/Ethics committee approval process steps need expansion.",
                suggestion="Include specific documentation requirements and submission timelines."
            )
        ]
        
        metadata = {
            "analysis_timestamp": datetime.now().isoformat(),
            "text_length": len(request.text),
            "model_version": "1.0.0"
        }
        
        response = AnalysisResponse(
            **scores,
            issues=issues,
            metadata=metadata
        )
        
        logger.info(f"Returning analysis with {len(issues)} issues")
        return response
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Ilana Protocol Intelligence API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)