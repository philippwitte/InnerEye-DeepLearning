#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, Optional

import nbformat
import papermill
from IPython.display import Markdown, display
from nbconvert import HTMLExporter
from nbconvert.writers import FilesWriter

from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import FULL_METRICS_DATAFRAME_FILE, METRICS_AGGREGATES_FILE


def print_header(message: str, level: int = 2) -> None:
    """
    Displays a message, and afterwards repeat it as Markdown with the given indentation level (level=1 is the
    outermost, `# Foo`.
    :param message: The header string to display.
    :param level: The Markdown indentation level. level=1 for top level, level=3 for `### Foo`
    """
    prefix = "#" * level
    display(Markdown(f"{prefix} {message}"))


def generate_notebook(template_notebook: Path, notebook_params: Dict, result_notebook: Path) -> Path:
    print(f"Writing report to {result_notebook}")
    papermill.execute_notebook(input_path=str(template_notebook),
                               output_path=str(result_notebook),
                               parameters=notebook_params,
                               progress_bar=False)
    print(f"Running conversion to HTML for {result_notebook}")
    with result_notebook.open() as f:
        notebook = nbformat.read(f, as_version=4)
        html_exporter = HTMLExporter()
        html_exporter.exclude_input = True
        (body, resources) = html_exporter.from_notebook_node(notebook)
        write_file = FilesWriter()
        write_file.build_directory = str(result_notebook.parent)
        write_file.write(
            output=body,
            resources=resources,
            notebook_name=str(result_notebook.stem)
        )
    return result_notebook.with_suffix(resources['output_extension'])


def generate_segmentation_notebook(result_notebook: Path,
                                   train_metrics: Optional[Path] = None,
                                   val_metrics: Optional[Path] = None,
                                   test_metrics: Optional[Path] = None) -> Path:
    """
    Creates a reporting notebook for a segmentation model, using the given training, validation, and test set metrics.
    Returns the report file after HTML conversion.
    """
    def str_or_empty(p: Optional[Path]) -> str:
        return str(p) if p else ""

    notebook_params = \
        {
            'innereye_path': str(fixed_paths.repository_root_directory()),
            'train_metrics_csv': str_or_empty(train_metrics),
            'val_metrics_csv': str_or_empty(val_metrics),
            'test_metrics_csv': str_or_empty(test_metrics),
        }
    template = Path(__file__).absolute().parent / "segmentation_report.ipynb"
    return generate_notebook(template,
                             notebook_params=notebook_params,
                             result_notebook=result_notebook)


def generate_classification_notebook(result_notebook: Path, data_folder: Optional[Path] = None):
    def str_or_empty(p: Optional[Path]) -> str:
        return str(p) if p else ""

    metrics_aggregates_file = data_folder / METRICS_AGGREGATES_FILE
    metrics_across_all_runs_file = data_folder / FULL_METRICS_DATAFRAME_FILE
    notebook_params = \
        {
            'metrics_aggregates_file': str(metrics_aggregates_file),
            'metrics_across_all_runs_file': str(metrics_across_all_runs_file)
        }

    template = Path(__file__).absolute().parent / "classification_report.ipynb"
    return generate_notebook(template,
                             notebook_params=notebook_params,
                             result_notebook=result_notebook)
