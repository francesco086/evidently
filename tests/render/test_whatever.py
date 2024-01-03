from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import numpy as np
import pandas as pd
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report
from pytest import fixture
from sklearn import datasets

# DataQualityPreset.__name__,
# ClassificationPreset.__name__,
# RegressionPreset.__name__,
# TargetDriftPreset.__name__,
# TextOverviewPreset.__name__,
# RecsysPreset.__name__,


def test_data_drift_preset(data: Tuple[pd.DataFrame, pd.DataFrame], tmp_dir: Path) -> None:
    adult_ref, adult_cur = data
    for preset in [DataDriftPreset()]:
        report = Report(metrics=[preset])

        report.run(reference_data=adult_ref, current_data=adult_cur)
        report.save_html((tmp_dir / "data_stability.html").as_posix())

        assert "MyReferee" in (tmp_dir / "data_stability.html").read_text()
        assert "MyCorrente" in (tmp_dir / "data_stability.html").read_text()
        assert "Reference " not in (tmp_dir / "data_stability.html").read_text()
        assert "Current " not in (tmp_dir / "data_stability.html").read_text()
        
        
def test_data_quality_preset(data: Tuple[pd.DataFrame, pd.DataFrame], tmp_dir: Path) -> None:
    adult_ref, adult_cur = data
    for preset in [DataQualityPreset()]:
        report = Report(metrics=[preset])

        report.run(reference_data=adult_ref, current_data=adult_cur)
        report.save_html((tmp_dir / "data_stability.html").as_posix())


        assert "MyReferee" in (tmp_dir / "data_stability.html").read_text()
        assert "MyCorrente" in (tmp_dir / "data_stability.html").read_text()
        assert "Reference " not in (tmp_dir / "data_stability.html").read_text()
        assert "Current " not in (tmp_dir / "data_stability.html").read_text()


@fixture(scope="module")
def data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    adult_data = datasets.fetch_openml(
        name="adult", version=2, as_frame="auto", parser="auto"
    )
    adult = adult_data.frame

    adult_ref = adult[~adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]
    adult_cur = adult[adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]

    adult_cur.iloc[:2000, 3:5] = np.nan
    return adult_ref, adult_cur


@fixture
def tmp_dir() -> Path:
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
