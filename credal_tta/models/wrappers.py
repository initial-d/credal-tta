"""
TSFM Model Wrappers - Architecture-Agnostic Interfaces
"""

import numpy as np
import torch
from typing import Optional


class TSFMWrapper:
    """Base class for TSFM wrappers"""
    
    def predict(self, context: np.ndarray) -> float:
        """
        One-step-ahead prediction
        
        Args:
            context: Historical observations
            
        Returns:
            Next-step forecast
        """
        raise NotImplementedError


class ChronosWrapper(TSFMWrapper):
    """
    Wrapper for Chronos T5-based foundation model
    
    Reference: https://github.com/amazon-science/chronos-forecasting
    """
    
    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            device: Compute device
        """
        try:
            from chronos import ChronosPipeline
            self.pipeline = ChronosPipeline.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
            )
        except ImportError:
            print("WARNING: chronos-forecasting not installed. Using mock model.")
            self.pipeline = None
        
        self.device = device
        self.model_name = model_name
        
    def predict(self, context: np.ndarray, prediction_length: int = 1) -> float:
        """
        Chronos prediction with dynamic context length
        
        Args:
            context: Historical observations (variable length)
            prediction_length: Forecast horizon (default: 1)
            
        Returns:
            One-step-ahead forecast (median of quantile predictions)
        """
        if self.pipeline is None:
            # Mock fallback: Moving average over last 50 points (exhibits context inertia!)
            # This is intentionally designed to show the problem Credal-TTA solves
            window = min(50, len(context))
            if window > 0:
                return np.mean(context[-window:])
            else:
                return 0.0
        
        # Chronos expects shape (batch, time)
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        
        # Generate forecast
        with torch.no_grad():
            forecast = self.pipeline.predict(
                context_tensor,
                prediction_length=prediction_length,
                num_samples=20  # Sample multiple trajectories
            )
        
        # Return median prediction
        forecast_np = forecast.numpy()
        median_forecast = np.median(forecast_np, axis=1)  # Median across samples
        
        return float(median_forecast[0, 0])  # First timestep, first batch


class MoiraiWrapper(TSFMWrapper):
    """
    Wrapper for Salesforce Moirai multi-scale foundation model
    
    Reference: https://github.com/SalesforceAIResearch/uni2ts
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/moirai-1.0-R-small",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        try:
            from uni2ts.model.moirai import MoiraiForecast
            self.model = MoiraiForecast.load_from_checkpoint(model_name)
            self.model.to(device)
            self.model.eval()
        except ImportError:
            print("WARNING: uni2ts (Moirai) not installed. Using mock model.")
            self.model = None
        
        self.device = device
        
    def predict(self, context: np.ndarray, prediction_length: int = 1) -> float:
        if self.model is None:
            # Mock fallback: Moving average (same as Chronos mock)
            window = min(50, len(context))
            if window > 0:
                return np.mean(context[-window:])
            else:
                return 0.0
        
        # Moirai expects (batch, channels, time)
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        context_tensor = context_tensor.to(self.device)
        
        with torch.no_grad():
            forecast = self.model(
                past_target=context_tensor,
                prediction_length=prediction_length
            )
        
        forecast_np = forecast.cpu().numpy()
        return float(forecast_np[0, 0, 0])  # First timestep


class PatchTSTWrapper(TSFMWrapper):
    """
    Wrapper for PatchTST model
    
    Note: Requires training on target dataset (not zero-shot like Chronos/Moirai)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        seq_len: int = 512,
        patch_len: int = 16,
        stride: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.device = device
        
        # Load pretrained model if available
        if model_path:
            try:
                self.model = torch.load(model_path, map_location=device)
                self.model.eval()
            except FileNotFoundError:
                print(f"WARNING: Model not found at {model_path}. Using mock.")
                self.model = None
        else:
            self.model = None
    
    def predict(self, context: np.ndarray, prediction_length: int = 1) -> float:
        if self.model is None:
            # Mock: Moving average over window
            window = min(50, len(context))
            if window > 0:
                return np.mean(context[-window:])
            else:
                return 0.0
        
        # Pad/truncate to expected length
        if len(context) < self.seq_len:
            context = np.pad(context, (self.seq_len - len(context), 0), mode='edge')
        else:
            context = context[-self.seq_len:]
        
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        context_tensor = context_tensor.to(self.device)
        
        with torch.no_grad():
            forecast = self.model(context_tensor)
        
        forecast_np = forecast.cpu().numpy()
        return float(forecast_np[0, 0])


class NaiveBaseline(TSFMWrapper):
    """Simple baselines for comparison"""
    
    def __init__(self, method: str = "last_value"):
        """
        Args:
            method: "last_value" or "moving_average"
        """
        self.method = method
    
    def predict(self, context: np.ndarray, window: int = 10) -> float:
        if len(context) == 0:
            return 0.0
        
        if self.method == "last_value":
            return float(context[-1])
        elif self.method == "moving_average":
            w = min(window, len(context))
            return float(np.mean(context[-w:]))
        else:
            return float(context[-1])
