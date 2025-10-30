from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime
from ai_service import IlanaAIService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AI Service
ai_service = IlanaAIService()

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
        
        # **REAL AI INTEGRATION** - Using Pinecone + PubmedBERT + Azure ML
        analysis_result = await ai_service.analyze_protocol_comprehensive(request.text)
        
        # Convert issues to Pydantic models
        issues = [
            Issue(
                type=issue["type"],
                message=issue["message"],
                suggestion=issue.get("suggestion")
            )
            for issue in analysis_result["issues"]
        ]
        
        response = AnalysisResponse(
            compliance_score=analysis_result["compliance_score"],
            clarity_score=analysis_result["clarity_score"],
            engagement_score=analysis_result["engagement_score"],
            delivery_score=analysis_result["delivery_score"],
            issues=issues,
            metadata=analysis_result["metadata"]
        )
        
        logger.info(f"✅ REAL AI ANALYSIS COMPLETE - {len(issues)} issues identified")
        logger.info(f"Scores: C={response.compliance_score}, Cl={response.clarity_score}, E={response.engagement_score}, D={response.delivery_score}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Real AI analysis error: {str(e)}")
        # Fallback to basic analysis if AI service fails
        fallback_response = AnalysisResponse(
            compliance_score=75,
            clarity_score=72,
            engagement_score=70,
            delivery_score=73,
            issues=[
                Issue(
                    type="system",
                    message="AI analysis temporarily unavailable - using fallback mode",
                    suggestion="Please try again in a few moments"
                )
            ],
            metadata={
                "analysis_timestamp": datetime.now().isoformat(),
                "text_length": len(request.text),
                "model_version": "fallback-1.0.0",
                "error": str(e)
            }
        )
        return fallback_response

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Ilana Protocol Intelligence API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)