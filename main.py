from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import shutil
import os
from io import BytesIO
import pandas as pd
import uuid
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

import database
import schemas
import crud
import data_generator
import config
from utils import image_helpers
from utils import image_augmentor
from processing import classical, feature_based, cnn

database.create_db_and_tables()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(config.IMAGE_ACCESS_URL_PREFIX,
          StaticFiles(directory=config.DATA_STORAGE_PATH),
          name="stored_images")


def _process_image_core(
        image_np: np.ndarray,
        processing_params: schemas.ProcessBaseRequest,
        db_experiment_data: dict
) -> dict:

    current_image_for_processing = image_np.copy()
    aug_details = None

    # -- 1
    if processing_params.augmentation and processing_params.augmentation.operation:
        aug_op = processing_params.augmentation.operation
        aug_param = processing_params.augmentation.parameter

        try:
            augmented_image_np = image_augmentor.apply_augmentation(
                current_image_for_processing, aug_op, aug_param
            )
            current_image_for_processing = augmented_image_np
            aug_details = {**processing_params.augmentation.model_dump()}
        except Exception as e:
            error_msg = f"Аугментация провалилась '{aug_op}': {str(e)}"
            print(error_msg) # TODO debug
            aug_details = {"error": error_msg, **processing_params.augmentation.model_dump()}

    db_experiment_data["augmentation_details"] = aug_details

    # -- 2
    if processing_params.method1.apply:
        method1_params = processing_params.method1.params or {}
        db_experiment_data["result_method1"] = classical.count_cells_classical(
            current_image_for_processing, method1_params
        )
        db_experiment_data["params_method1"] = method1_params

    if processing_params.method2.apply:
        method2_params = processing_params.method2.params or {}
        db_experiment_data["result_method2"] = feature_based.count_cells_feature_based(
            current_image_for_processing, method2_params
        )
        db_experiment_data["params_method2"] = method2_params

    if processing_params.method3.apply:
        method3_params = processing_params.method3.params or {}
        db_experiment_data["result_method3"] = cnn.count_cells_cnn(
            current_image_for_processing, method3_params
        )
        db_experiment_data["params_method3"] = method3_params

    return db_experiment_data

@app.post("/experiments/generate_and_process", response_model=schemas.ExperimentRecord, status_code=201)
async def generate_and_process_experiment(
        request_data: schemas.ProcessGeneratedImageRequest,
        db: Session = Depends(database.get_db)
):
    image_np, true_cell_count = data_generator.generate_synthetic_cells_image(request_data.generator_params)
    generated_filename, _ = image_helpers.save_image_to_path(
        image_np, config.DATA_STORAGE_PATH, "gen_"
    )
    db_experiment_data = {
        "data_source_type": "generated",
        "data_source_info": {
            "generator_params": request_data.generator_params.model_dump(),
            "true_cell_count": true_cell_count,
            "stored_filename": generated_filename
        }
    }
    completed_experiment_data = _process_image_core(image_np, request_data, db_experiment_data)
    experiment_to_create = schemas.ExperimentCreate(**completed_experiment_data)
    created_experiment = crud.create_experiment(db, experiment_to_create)
    return created_experiment


@app.post("/experiments/upload_and_process", response_model=schemas.ExperimentRecord, status_code=201)
async def upload_and_process_experiment(
        file: UploadFile = File(...),
        augmentation_json: Optional[str] = Form(None),
        method1_json: Optional[str] = Form(None),
        method2_json: Optional[str] = Form(None),
        method3_json: Optional[str] = Form(None),
        db: Session = Depends(database.get_db)
):
    original_filename = file.filename
    stored_filename, stored_filepath = image_helpers.save_upload_file_to_storage(
        file, config.DATA_STORAGE_PATH
    )
    try:
        image_np = image_helpers.read_image_from_path(stored_filepath)
        if image_np is None:
            raise HTTPException(status_code=400, detail="Не удалось прочитать файл.")
    except Exception as e:
        if stored_filepath.exists():
            os.remove(stored_filepath)
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

    def parse_json_form_data(json_str: Optional[str], model_type: type):
        if not json_str:
            return model_type() if model_type == schemas.MethodExecutionParams else None
        try:
            data = json.loads(json_str)
            return model_type(**data)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Неверный JSON для {model_type.__name__}: {json_str}. Error: {e}")

    processing_params = schemas.ProcessBaseRequest(
        augmentation=parse_json_form_data(augmentation_json, schemas.AugmentationParams),
        method1=parse_json_form_data(method1_json, schemas.MethodExecutionParams),
        method2=parse_json_form_data(method2_json, schemas.MethodExecutionParams),
        method3=parse_json_form_data(method3_json, schemas.MethodExecutionParams),
    )

    db_experiment_data = {
        "data_source_type": "real_file",
        "data_source_info": {
            "original_filename": original_filename,
            "stored_filename": stored_filename
        }
    }
    completed_experiment_data = _process_image_core(image_np, processing_params, db_experiment_data)
    experiment_to_create = schemas.ExperimentCreate(**completed_experiment_data)
    created_experiment = crud.create_experiment(db, experiment_to_create)
    return created_experiment


@app.get("/experiments", response_model=List[schemas.ExperimentRecord])
async def list_experiments(
        skip: int = 0,
        limit: int = Query(default=20, ge=1, le=100),
        db: Session = Depends(database.get_db)
):
    experiments = crud.get_experiments(db, skip=skip, limit=limit)
    return experiments

@app.get("/experiments/{experiment_id}", response_model=schemas.ExperimentRecord)
async def get_experiment_details(experiment_id: int, db: Session = Depends(database.get_db)):
    experiment = crud.get_experiment_by_id(db, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment

@app.get("/experiments/export/csv")
async def export_experiments_to_csv(db: Session = Depends(database.get_db)):
    experiments = crud.get_all_experiments_for_export(db)
    if not experiments:
        return JSONResponse(content={"message": "No experiments"}, status_code=404)

    exp_list = []
    for exp_obj in experiments:
        exp_dict = {c.name: getattr(exp_obj, c.name) for c in exp_obj.__table__.columns}

        ds_info = exp_dict.pop('data_source_info', {}) or {}
        if exp_dict.get('data_source_type') == 'generated':
            gen_params = ds_info.get('generator_params', {}) or {}
            for k, v in gen_params.items():
                exp_dict[f'gen_param_{k}'] = v
            exp_dict['gen_true_cell_count'] = ds_info.get('true_cell_count')
        exp_dict['ds_original_filename'] = ds_info.get('original_filename')
        exp_dict['ds_stored_filename'] = ds_info.get('stored_filename')

        aug_details = exp_dict.pop('augmentation_details', {}) or {}
        exp_dict['aug_operation'] = aug_details.get('operation')
        exp_dict['aug_parameter'] = aug_details.get('parameter')
        exp_dict['aug_error'] = aug_details.get('error')
        # exp_dict['aug_temp_filename'] = aug_details.get('augmented_temp_filename')
        exp_dict['aug_stored_filename'] = aug_details.get('stored_augmented_filename')

        exp_list.append(exp_dict)

    df = pd.DataFrame(exp_list)

    stream = BytesIO()
    df.to_csv(stream, index=False, encoding='utf-8')
    stream.seek(0)

    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cell_counting_experiments.csv"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
