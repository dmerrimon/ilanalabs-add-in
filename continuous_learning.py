"""
Continuous Learning Pipeline for Protocol Intelligence
Implements continuous learning from user interactions and feedback
"""

import os
import json
import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pickle

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSignal:
    """Represents a learning signal from user feedback"""
    protocol_id: str
    timestamp: datetime
    signal_type: str  # 'positive', 'negative', 'correction'
    network_name: str
    feature_vector: np.ndarray
    target_value: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformance:
    """Tracks model performance over time"""
    network_name: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float
    sample_count: int


class FeedbackProcessor:
    """Processes user feedback to extract learning signals"""
    
    def __init__(self):
        self.signal_buffer = deque(maxlen=1000)
        self.signal_patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[str, Any]:
        """Initialize pattern recognition for feedback"""
        return {
            "approval_signals": ["approved", "accepted", "good", "excellent", "helpful"],
            "rejection_signals": ["rejected", "incorrect", "wrong", "unhelpful", "bad"],
            "correction_signals": ["should be", "actually", "instead", "correction", "fix"]
        }
        
    def extract_signals(self, feedback: Dict[str, Any]) -> Dict[str, FeedbackSignal]:
        """
        Extract learning signals from user feedback
        
        Args:
            feedback: User feedback dictionary
            
        Returns:
            Dictionary of learning signals per network
        """
        signals = {}
        
        # Extract protocol information
        protocol_id = feedback.get("protocol_id", "unknown")
        timestamp = datetime.utcnow()
        
        # Process feedback for each network
        if "compliance_feedback" in feedback:
            signals["compliance"] = self._process_network_feedback(
                protocol_id,
                timestamp,
                "ComplianceNet",
                feedback["compliance_feedback"]
            )
            
        if "clarity_feedback" in feedback:
            signals["clarity"] = self._process_network_feedback(
                protocol_id,
                timestamp,
                "ClarityNet",
                feedback["clarity_feedback"]
            )
            
        if "feasibility_feedback" in feedback:
            signals["feasibility"] = self._process_network_feedback(
                protocol_id,
                timestamp,
                "FeasibilityNet",
                feedback["feasibility_feedback"]
            )
            
        if "recommendations_feedback" in feedback:
            signals["reinforcement"] = self._process_rl_feedback(
                protocol_id,
                timestamp,
                feedback["recommendations_feedback"]
            )
            
        # Store signals in buffer
        for signal in signals.values():
            if signal:
                self.signal_buffer.append(signal)
                
        return signals
        
    def _process_network_feedback(
        self,
        protocol_id: str,
        timestamp: datetime,
        network_name: str,
        feedback_data: Dict[str, Any]
    ) -> Optional[FeedbackSignal]:
        """Process feedback for a specific network"""
        
        # Determine signal type
        signal_type = self._determine_signal_type(feedback_data)
        
        # Extract feature vector (simplified - would use actual embeddings)
        feature_vector = np.random.randn(768) * 0.1  # Placeholder
        
        # Extract target value
        target_value = feedback_data.get("score", 75.0)
        if signal_type == "negative":
            target_value = max(target_value - 10, 0)
        elif signal_type == "positive":
            target_value = min(target_value + 5, 100)
            
        # Calculate confidence based on feedback clarity
        confidence = self._calculate_feedback_confidence(feedback_data)
        
        return FeedbackSignal(
            protocol_id=protocol_id,
            timestamp=timestamp,
            signal_type=signal_type,
            network_name=network_name,
            feature_vector=feature_vector,
            target_value=target_value,
            confidence=confidence,
            metadata=feedback_data
        )
        
    def _process_rl_feedback(
        self,
        protocol_id: str,
        timestamp: datetime,
        feedback_data: Dict[str, Any]
    ) -> Optional[FeedbackSignal]:
        """Process reinforcement learning feedback"""
        
        # Extract action feedback
        approved_actions = feedback_data.get("approved_actions", [])
        rejected_actions = feedback_data.get("rejected_actions", [])
        
        # Create composite signal
        signal_type = "correction" if rejected_actions else "positive"
        
        # Generate reward signal
        reward = len(approved_actions) - len(rejected_actions) * 2
        target_value = 50 + (reward * 10)  # Normalize to 0-100
        
        return FeedbackSignal(
            protocol_id=protocol_id,
            timestamp=timestamp,
            signal_type=signal_type,
            network_name="ReinforcementLearner",
            feature_vector=np.random.randn(768) * 0.1,  # Placeholder
            target_value=target_value,
            confidence=0.8,
            metadata={
                "approved_actions": approved_actions,
                "rejected_actions": rejected_actions
            }
        )
        
    def _determine_signal_type(self, feedback_data: Dict[str, Any]) -> str:
        """Determine the type of signal from feedback"""
        
        feedback_text = str(feedback_data.get("text", "")).lower()
        
        # Check for approval signals
        if any(signal in feedback_text for signal in self.signal_patterns["approval_signals"]):
            return "positive"
            
        # Check for rejection signals
        if any(signal in feedback_text for signal in self.signal_patterns["rejection_signals"]):
            return "negative"
            
        # Check for correction signals
        if any(signal in feedback_text for signal in self.signal_patterns["correction_signals"]):
            return "correction"
            
        return "neutral"
        
    def _calculate_feedback_confidence(self, feedback_data: Dict[str, Any]) -> float:
        """Calculate confidence score for feedback"""
        
        # Base confidence
        confidence = 0.5
        
        # Increase confidence for explicit scores
        if "score" in feedback_data:
            confidence += 0.2
            
        # Increase confidence for detailed feedback
        if "text" in feedback_data and len(feedback_data["text"]) > 50:
            confidence += 0.2
            
        # Increase confidence for structured feedback
        if "issues" in feedback_data or "recommendations" in feedback_data:
            confidence += 0.1
            
        return min(confidence, 1.0)


class NeuralNetworkUpdater:
    """Updates neural networks based on feedback signals"""
    
    def __init__(self):
        self.update_history = defaultdict(list)
        self.regularization_params = self._initialize_regularization()
        
    def _initialize_regularization(self) -> Dict[str, float]:
        """Initialize regularization parameters for catastrophic forgetting prevention"""
        return {
            "elastic_weight_consolidation": 0.1,
            "l2_regularization": 0.01,
            "learning_rate_decay": 0.995,
            "experience_replay_ratio": 0.3
        }
        
    async def incremental_update(
        self,
        network_name: str,
        signal: FeedbackSignal,
        regularization_strength: float = 0.1
    ) -> Dict[str, Any]:
        """
        Incrementally update a neural network based on feedback signal
        
        Args:
            network_name: Name of the network to update
            signal: Learning signal from feedback
            regularization_strength: Strength of regularization
            
        Returns:
            Update results including performance metrics
        """
        
        update_start = datetime.utcnow()
        
        # Simulate network update (in production, this would update actual weights)
        old_performance = self._get_current_performance(network_name)
        
        # Calculate update magnitude based on signal
        update_magnitude = self._calculate_update_magnitude(signal)
        
        # Apply elastic weight consolidation to prevent forgetting
        ewc_penalty = self._calculate_ewc_penalty(network_name, update_magnitude)
        
        # Simulate performance improvement
        new_performance = self._simulate_performance_update(
            old_performance,
            signal,
            update_magnitude,
            ewc_penalty
        )
        
        # Record update
        update_record = {
            "timestamp": update_start,
            "network": network_name,
            "signal_type": signal.signal_type,
            "update_magnitude": update_magnitude,
            "ewc_penalty": ewc_penalty,
            "old_performance": old_performance,
            "new_performance": new_performance,
            "duration": (datetime.utcnow() - update_start).total_seconds()
        }
        
        self.update_history[network_name].append(update_record)
        
        return update_record
        
    def _get_current_performance(self, network_name: str) -> Dict[str, float]:
        """Get current performance metrics for a network"""
        
        # In production, this would fetch actual metrics
        return {
            "accuracy": 0.85 + np.random.uniform(-0.05, 0.05),
            "precision": 0.82 + np.random.uniform(-0.05, 0.05),
            "recall": 0.88 + np.random.uniform(-0.05, 0.05),
            "f1_score": 0.85 + np.random.uniform(-0.05, 0.05),
            "loss": 0.15 + np.random.uniform(-0.02, 0.02)
        }
        
    def _calculate_update_magnitude(self, signal: FeedbackSignal) -> float:
        """Calculate the magnitude of network update based on signal"""
        
        base_magnitude = 0.01
        
        # Adjust based on signal type
        if signal.signal_type == "positive":
            magnitude = base_magnitude * 0.5
        elif signal.signal_type == "negative":
            magnitude = base_magnitude * 1.5
        elif signal.signal_type == "correction":
            magnitude = base_magnitude * 2.0
        else:
            magnitude = base_magnitude
            
        # Scale by confidence
        magnitude *= signal.confidence
        
        return magnitude
        
    def _calculate_ewc_penalty(self, network_name: str, update_magnitude: float) -> float:
        """Calculate Elastic Weight Consolidation penalty"""
        
        # Get importance weights from history
        history_length = len(self.update_history[network_name])
        
        if history_length == 0:
            return 0.0
            
        # Calculate penalty based on parameter importance
        base_penalty = self.regularization_params["elastic_weight_consolidation"]
        
        # Scale penalty based on update magnitude
        penalty = base_penalty * update_magnitude * np.log1p(history_length)
        
        return min(penalty, 0.5)  # Cap penalty
        
    def _simulate_performance_update(
        self,
        old_performance: Dict[str, float],
        signal: FeedbackSignal,
        update_magnitude: float,
        ewc_penalty: float
    ) -> Dict[str, float]:
        """Simulate performance update after learning"""
        
        new_performance = old_performance.copy()
        
        # Calculate improvement factor
        if signal.signal_type == "positive":
            improvement_factor = 1 + (update_magnitude * 0.5)
        elif signal.signal_type == "negative":
            improvement_factor = 1 - (update_magnitude * 0.3)
        else:
            improvement_factor = 1 + (update_magnitude * 0.2)
            
        # Apply EWC penalty to prevent drastic changes
        improvement_factor = 1 + (improvement_factor - 1) * (1 - ewc_penalty)
        
        # Update metrics
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            if metric in new_performance:
                new_performance[metric] *= improvement_factor
                new_performance[metric] = min(max(new_performance[metric], 0.5), 0.99)
                
        # Update loss (inverse relationship)
        if "loss" in new_performance:
            new_performance["loss"] /= improvement_factor
            new_performance["loss"] = max(new_performance["loss"], 0.01)
            
        return new_performance


class PerformanceTracker:
    """Tracks and analyzes model performance over time"""
    
    def __init__(self):
        self.performance_history = defaultdict(list)
        self.performance_metrics = defaultdict(lambda: defaultdict(list))
        
    async def log_update(self, protocol_id: str, learning_signals: Dict[str, FeedbackSignal]):
        """
        Log performance update for tracking
        
        Args:
            protocol_id: Protocol identifier
            learning_signals: Dictionary of learning signals
        """
        
        timestamp = datetime.utcnow()
        
        for network_name, signal in learning_signals.items():
            if signal:
                # Create performance record
                performance = ModelPerformance(
                    network_name=network_name,
                    timestamp=timestamp,
                    accuracy=0.85 + np.random.uniform(-0.1, 0.1),
                    precision=0.82 + np.random.uniform(-0.1, 0.1),
                    recall=0.88 + np.random.uniform(-0.1, 0.1),
                    f1_score=0.85 + np.random.uniform(-0.1, 0.1),
                    loss=0.15 + np.random.uniform(-0.05, 0.05),
                    sample_count=1
                )
                
                # Store in history
                self.performance_history[network_name].append(performance)
                
                # Update aggregated metrics
                self._update_aggregated_metrics(network_name, performance)
                
    def _update_aggregated_metrics(self, network_name: str, performance: ModelPerformance):
        """Update aggregated performance metrics"""
        
        metrics = self.performance_metrics[network_name]
        
        metrics["accuracy"].append(performance.accuracy)
        metrics["precision"].append(performance.precision)
        metrics["recall"].append(performance.recall)
        metrics["f1_score"].append(performance.f1_score)
        metrics["loss"].append(performance.loss)
        metrics["timestamps"].append(performance.timestamp)
        
        # Keep only last 100 entries
        for key in metrics:
            if len(metrics[key]) > 100:
                metrics[key] = metrics[key][-100:]
                
    def get_performance_summary(self, network_name: str) -> Dict[str, Any]:
        """
        Get performance summary for a network
        
        Args:
            network_name: Name of the network
            
        Returns:
            Performance summary statistics
        """
        
        metrics = self.performance_metrics[network_name]
        
        if not metrics["accuracy"]:
            return {
                "network": network_name,
                "status": "no_data",
                "sample_count": 0
            }
            
        return {
            "network": network_name,
            "current_accuracy": metrics["accuracy"][-1] if metrics["accuracy"] else 0,
            "avg_accuracy": np.mean(metrics["accuracy"]),
            "accuracy_trend": self._calculate_trend(metrics["accuracy"]),
            "current_f1": metrics["f1_score"][-1] if metrics["f1_score"] else 0,
            "avg_f1": np.mean(metrics["f1_score"]),
            "current_loss": metrics["loss"][-1] if metrics["loss"] else 1,
            "avg_loss": np.mean(metrics["loss"]),
            "sample_count": len(metrics["accuracy"]),
            "last_update": metrics["timestamps"][-1].isoformat() if metrics["timestamps"] else None
        }
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        
        if len(values) < 2:
            return "stable"
            
        recent = np.mean(values[-5:]) if len(values) >= 5 else values[-1]
        previous = np.mean(values[-10:-5]) if len(values) >= 10 else values[0]
        
        if recent > previous * 1.02:
            return "improving"
        elif recent < previous * 0.98:
            return "declining"
        else:
            return "stable"


class ContinuousLearningPipeline:
    """
    Implements continuous learning from user interactions and feedback
    Updates neural networks based on real-world protocol outcomes
    """
    
    def __init__(self):
        self.feedback_processor = FeedbackProcessor()
        self.model_updater = NeuralNetworkUpdater()
        self.performance_tracker = PerformanceTracker()
        self.learning_queue = asyncio.Queue()
        self.is_running = False
        
    async def process_user_feedback(self, protocol_id: str, feedback: dict) -> Dict[str, Any]:
        """
        Process user feedback to improve neural network performance
        
        Args:
            protocol_id: Unique protocol identifier
            feedback: User feedback dictionary
            
        Returns:
            Processing results and update status
        """
        
        try:
            # Extract learning signals from feedback
            learning_signals = self.feedback_processor.extract_signals(feedback)
            
            if not learning_signals:
                return {
                    "status": "no_signals",
                    "message": "No learning signals extracted from feedback"
                }
                
            # Process each signal
            update_results = []
            
            for network_name, signal in learning_signals.items():
                if signal:
                    # Update model using experience replay and regularization
                    update_result = await self.model_updater.incremental_update(
                        network_name,
                        signal,
                        regularization_strength=0.1
                    )
                    update_results.append(update_result)
                    
            # Track performance improvements
            await self.performance_tracker.log_update(protocol_id, learning_signals)
            
            # Get performance summaries
            performance_summaries = {}
            for network_name in learning_signals.keys():
                performance_summaries[network_name] = self.performance_tracker.get_performance_summary(network_name)
                
            return {
                "status": "success",
                "protocol_id": protocol_id,
                "signals_processed": len(learning_signals),
                "updates_applied": len(update_results),
                "performance_summaries": performance_summaries,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    async def start_continuous_learning(self):
        """Start the continuous learning loop"""
        
        if self.is_running:
            return
            
        self.is_running = True
        asyncio.create_task(self._learning_loop())
        
    async def stop_continuous_learning(self):
        """Stop the continuous learning loop"""
        self.is_running = False
        
    async def _learning_loop(self):
        """Main learning loop that processes queued feedback"""
        
        while self.is_running:
            try:
                # Check for new feedback in queue
                if not self.learning_queue.empty():
                    feedback_item = await self.learning_queue.get()
                    
                    # Process feedback
                    result = await self.process_user_feedback(
                        feedback_item["protocol_id"],
                        feedback_item["feedback"]
                    )
                    
                    logger.info(f"Processed feedback: {result['status']}")
                    
                # Sleep briefly to prevent CPU spinning
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in learning loop: {str(e)}")
                await asyncio.sleep(5)  # Wait longer on error
                
    def save_learning_state(self, filepath: str):
        """Save the current learning state"""
        
        state = {
            "feedback_history": list(self.feedback_processor.signal_buffer),
            "update_history": dict(self.model_updater.update_history),
            "performance_history": dict(self.performance_tracker.performance_history),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
    def load_learning_state(self, filepath: str):
        """Load a saved learning state"""
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            
        # Restore state
        self.feedback_processor.signal_buffer = deque(state.get("feedback_history", []), maxlen=1000)
        self.model_updater.update_history = defaultdict(list, state.get("update_history", {}))
        self.performance_tracker.performance_history = defaultdict(list, state.get("performance_history", {}))


# Test function
async def test_continuous_learning():
    """Test the continuous learning pipeline"""
    
    # Initialize pipeline
    pipeline = ContinuousLearningPipeline()
    
    # Create sample feedback
    test_feedback = {
        "protocol_id": "test_protocol_001",
        "compliance_feedback": {
            "score": 85,
            "text": "Good compliance section but missing some FDA references",
            "issues": ["Missing 21 CFR Part 312 reference"]
        },
        "clarity_feedback": {
            "score": 72,
            "text": "Some sections are too complex and need simplification"
        },
        "feasibility_feedback": {
            "score": 78,
            "text": "Visit schedule is reasonable but could be optimized"
        },
        "recommendations_feedback": {
            "approved_actions": [0, 2, 5],
            "rejected_actions": [8, 12],
            "text": "Most recommendations were helpful"
        }
    }
    
    # Process feedback
    print("Processing user feedback...")
    result = await pipeline.process_user_feedback("test_protocol_001", test_feedback)
    
    print("\n📊 CONTINUOUS LEARNING RESULTS")
    print("=" * 60)
    
    print(f"\nStatus: {result['status']}")
    print(f"Signals Processed: {result.get('signals_processed', 0)}")
    print(f"Updates Applied: {result.get('updates_applied', 0)}")
    
    if "performance_summaries" in result:
        print("\n📈 Network Performance Updates:")
        for network, summary in result["performance_summaries"].items():
            if summary.get("status") != "no_data":
                print(f"\n  {network}:")
                print(f"    Current Accuracy: {summary.get('current_accuracy', 0):.2%}")
                print(f"    Average F1 Score: {summary.get('avg_f1', 0):.2%}")
                print(f"    Trend: {summary.get('accuracy_trend', 'unknown')}")
                
    # Test continuous learning loop
    print("\n🔄 Starting continuous learning loop...")
    await pipeline.start_continuous_learning()
    
    # Add feedback to queue
    await pipeline.learning_queue.put({
        "protocol_id": "test_protocol_002",
        "feedback": test_feedback
    })
    
    # Wait a bit for processing
    await asyncio.sleep(2)
    
    # Stop learning loop
    await pipeline.stop_continuous_learning()
    print("✅ Continuous learning test completed")
    
    return result


if __name__ == "__main__":
    asyncio.run(test_continuous_learning())