from typing import Optional, List

from sqlalchemy.orm import Session
import database, schemas

def create_experiment(db: Session, experiment: schemas.ExperimentCreate) -> database.Experiment:
    db_experiment = database.Experiment(
        data_source_type=experiment.data_source_type,
        data_source_info=experiment.data_source_info,
        augmentation_details=experiment.augmentation_details,
        result_method1=experiment.result_method1,
        params_method1=experiment.params_method1,
        result_method2=experiment.result_method2,
        params_method2=experiment.params_method2,
        result_method3=experiment.result_method3,
        params_method3=experiment.params_method3,
    )
    db.add(db_experiment)
    db.commit()
    db.refresh(db_experiment)
    return db_experiment

def get_experiment_by_id(db: Session, experiment_id: int) -> Optional[database.Experiment]:
    return db.query(database.Experiment).filter(database.Experiment.id == experiment_id).first()

def get_experiments(db: Session, skip: int = 0, limit: int = 100) -> List[database.Experiment]:
    return db.query(database.Experiment).order_by(database.Experiment.timestamp.desc()).offset(skip).limit(limit).all()

def get_all_experiments_for_export(db: Session) -> List[database.Experiment]:
    return db.query(database.Experiment).order_by(database.Experiment.timestamp.asc()).all()
