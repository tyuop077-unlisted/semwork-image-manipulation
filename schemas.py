from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import datetime

class GeneratorParams(BaseModel):
    num_cells: int = Field(10, gt=0, le=1000)
    min_radius: int = Field(5, gt=0)
    max_radius: int = Field(20, gt=0)
    image_width: int = Field(256, gt=50, le=2048)
    image_height: int = Field(256, gt=50, le=2048)
    overlap: bool = True
    noise_level: float = Field(0.0, ge=0.0, le=1.0) # 0 до 1 fraction

class AugmentationParams(BaseModel):
    operation: str
    parameter: Optional[int] = None

class ExperimentBase(BaseModel):
    data_source_type: str
    data_source_info: Dict[str, Any]
    augmentation_details: Optional[Dict[str, Any]] = None
    params_method1: Optional[Dict[str, Any]] = None
    params_method2: Optional[Dict[str, Any]] = None
    params_method3: Optional[Dict[str, Any]] = None

class ExperimentCreate(ExperimentBase):
    result_method1: Optional[int] = None
    result_method2: Optional[int] = None
    result_method3: Optional[int] = None

class ExperimentRecord(ExperimentBase):
    id: int
    timestamp: datetime.datetime
    result_method1: Optional[int] = None
    result_method2: Optional[int] = None
    result_method3: Optional[int] = None

    class Config:
        from_attributes = True

# API
class MethodExecutionParams(BaseModel):
    apply: bool = False
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ProcessBaseRequest(BaseModel):
    augmentation: Optional[AugmentationParams] = None
    method1: MethodExecutionParams = Field(default_factory=MethodExecutionParams)
    method2: MethodExecutionParams = Field(default_factory=MethodExecutionParams)
    method3: MethodExecutionParams = Field(default_factory=MethodExecutionParams)

class ProcessGeneratedImageRequest(ProcessBaseRequest):
    generator_params: GeneratorParams
