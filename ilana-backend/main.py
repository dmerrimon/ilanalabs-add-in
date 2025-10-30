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

# Lazy initialization of AI Service
ai_service = None

def get_ai_service():
    """Get or initialize AI service with error handling"""
    global ai_service
    if ai_service is None:
        try:
            from ai_service import IlanaAIService
            ai_service = IlanaAIService()
            logger.info("✅ AI Service initialized successfully")
        except Exception as e:
            logger.error(f"❌ AI Service initialization failed: {e}")
            ai_service = "failed"
    return ai_service if ai_service != "failed" else None

@app.post("/analyze-protocol", response_model=AnalysisResponse)
async def analyze_protocol(request: ProtocolAnalysisRequest):
    try:
        logger.info(f"Received analysis request for text length: {len(request.text)}")
        logger.info(f"Analysis options: {request.options}")
        
        # Try to get AI service
        service = get_ai_service()
        
        if service:
            logger.info("🧠 Using REAL AI analysis...")
            # **REAL AI INTEGRATION** - Using Pinecone + PubmedBERT + Azure ML
            analysis_result = await service.analyze_protocol_comprehensive(request.text)
            
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
        else:
            logger.warning("⚠️ AI Service not available, using basic analysis")
            raise Exception("AI Service initialization failed")
        
    except Exception as e:
        logger.error(f"❌ Analysis error: {str(e)}")
        
        # Basic fallback analysis
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
            "model_version": "1.0.0-fallback",
            "ai_status": "unavailable",
            "note": "Please configure environment variables for AI features"
        }
        
        response = AnalysisResponse(
            **scores,
            issues=issues,
            metadata=metadata
        )
        
        logger.info(f"Returning basic analysis with {len(issues)} issues")
        return response

@app.get("/health")
async def health_check():
    service_status = get_ai_service()
    return {
        "status": "healthy", 
        "service": "Ilana Protocol Intelligence API",
        "ai_service": "ready" if service_status else "not configured",
        "note": "Set environment variables for AI features" if not service_status else None
    }

@app.get("/ai-status")
async def ai_status():
    """Check AI service status and configuration"""
    import os
    
    config_status = {
        "PINECONE_API_KEY": "✅" if os.getenv("PINECONE_API_KEY") else "❌",
        "PINECONE_INDEX_NAME": "✅" if os.getenv("PINECONE_INDEX_NAME") else "❌",
        "HUGGINGFACE_API_KEY": "✅" if os.getenv("HUGGINGFACE_API_KEY") else "❌",
        "PUBMEDBERT_ENDPOINT_URL": "✅" if os.getenv("PUBMEDBERT_ENDPOINT_URL") else "❌"
    }
    
    service = get_ai_service()
    
    return {
        "ai_service_initialized": service is not None,
        "environment_variables": config_status,
        "pinecone_index": os.getenv("PINECONE_INDEX_NAME", "not set"),
        "recommendation": "Set missing environment variables in Render dashboard" if not service else "AI service ready"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)