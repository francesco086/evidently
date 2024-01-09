from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator, Tuple
from evidently import ColumnMapping

import numpy as np
import pandas as pd
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    ClassificationPreset,
    RegressionPreset,
    TargetDriftPreset
)
from evidently.report import Report
from pytest import fixture
from sklearn import datasets

# TargetDriftPreset.__name__,
# TextOverviewPreset.__name__,
# RecsysPreset.__name__,


def test_data_drift_preset(openml_data: Tuple[pd.DataFrame, pd.DataFrame], tmp_dir: Path) -> None:
    adult_ref, adult_cur = openml_data
    report = Report(metrics=[DataDriftPreset()])

    report.run(reference_data=adult_ref, current_data=adult_cur)
    report.save_html((tmp_dir / "data_stability.html").as_posix())

    assert "MyReferee" in (tmp_dir / "data_stability.html").read_text()
    assert "MyCorrente" in (tmp_dir / "data_stability.html").read_text()
    assert "Reference " not in (tmp_dir / "data_stability.html").read_text()
    assert "Current " not in (tmp_dir / "data_stability.html").read_text()
        
        
def test_data_quality_preset(openml_data: Tuple[pd.DataFrame, pd.DataFrame], tmp_dir: Path) -> None:
    adult_ref, adult_cur = openml_data
    report = Report(metrics=[DataQualityPreset()])

    report.run(reference_data=adult_ref, current_data=adult_cur)
    report.save_html((tmp_dir / "data_stability.html").as_posix())


    assert "MyReferee" in (tmp_dir / "data_stability.html").read_text()
    assert "MyCorrente" in (tmp_dir / "data_stability.html").read_text()
    assert "Reference " not in (tmp_dir / "data_stability.html").read_text()
    assert "Current " not in (tmp_dir / "data_stability.html").read_text()
        
        
def test_regression_preset(openml_data: Tuple[pd.DataFrame, pd.DataFrame], tmp_dir: Path) -> None:
    adult_ref, adult_cur = openml_data
    report = Report(metrics=[RegressionPreset()])
    
    column_mapping = ColumnMapping()
    column_mapping.target = 'hours-per-week'
    column_mapping.prediction = 'random-hours-per-week'

    report.run(reference_data=adult_ref, current_data=adult_cur, column_mapping=column_mapping)
    report.save_html((tmp_dir / "data_stability.html").as_posix())

    assert "MyReferee" in (tmp_dir / "data_stability.html").read_text()
    assert "MyCorrente" in (tmp_dir / "data_stability.html").read_text()
    assert "Reference " not in (tmp_dir / "data_stability.html").read_text()
    assert "Current " not in (tmp_dir / "data_stability.html").read_text()
    
    
def test_classification_preset(openml_data: Tuple[pd.DataFrame, pd.DataFrame], tmp_dir: Path) -> None:
    adult_ref, adult_cur = openml_data
    report = Report(metrics=[ClassificationPreset()])
    
    column_mapping = ColumnMapping()
    column_mapping.target = 'hours-per-week'
    column_mapping.prediction = 'random-hours-per-week'

    report.run(reference_data=adult_ref, current_data=adult_cur, column_mapping=column_mapping)
    report.save_html((tmp_dir / "data_stability.html").as_posix())

    assert "MyReferee" in (tmp_dir / "data_stability.html").read_text()
    assert "MyCorrente" in (tmp_dir / "data_stability.html").read_text()
    assert "Reference " not in (tmp_dir / "data_stability.html").read_text()
    assert "Current " not in (tmp_dir / "data_stability.html").read_text()


def test_target_drift_preset(openml_data: Tuple[pd.DataFrame, pd.DataFrame], tmp_dir: Path) -> None:
    adult_ref, adult_cur = openml_data
    report = Report(metrics=[TargetDriftPreset()])
    
    column_mapping = ColumnMapping()
    column_mapping.target = 'hours-per-week'

    report.run(reference_data=adult_ref, current_data=adult_cur, column_mapping=column_mapping)
    report.save_html((tmp_dir / "data_stability.html").as_posix())

    assert "MyReferee" in (tmp_dir / "data_stability.html").read_text()
    assert "MyCorrente" in (tmp_dir / "data_stability.html").read_text()
    assert "Reference " not in (tmp_dir / "data_stability.html").read_text()
    assert "Current " not in (tmp_dir / "data_stability.html").read_text()


@fixture(scope="module")
def openml_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    adult_data = datasets.fetch_openml(
        name="adult", version=2, as_frame="auto", parser="auto"
    )
    adult = adult_data.frame
    
    adult["random-hours-per-week"] = adult["hours-per-week"].sample(frac=1).values

    adult_ref = adult[~adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]
    adult_cur = adult[adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]

    adult_cur.iloc[:2000, 3:5] = np.nan
    return adult_ref, adult_cur


@fixture
def tmp_dir() -> Generator[Path, None, None]:
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
